#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import argparse
import random
import yaml
import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmdet.utils import register_all_modules

# ==============================================================================
# 1. 强制环境确定性设置
# ==============================================================================
# 针对 CUDA >= 10.2，保证卷积算法确定性
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "42"

def set_deterministic_context(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 牺牲一点速度换取确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 禁止 tf32 (Ampere架构显卡上可能导致数值差异)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    print(f"[Setup] Deterministic mode enabled. Seed: {seed}")

# ==============================================================================
# 2. 基础路径与辅助工具
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MMDET = os.path.join(CURRENT_DIR, '../mmdetection')
if os.path.exists(LOCAL_MMDET):
    MMDET_ROOT = LOCAL_MMDET
else:
    MMDET_ROOT = '/data/application/wangwl/Detection/mmdetection'

def load_model_yaml(model_name):
    yaml_file = os.path.join(CURRENT_DIR, f"{model_name}.yml")
    if not os.path.exists(yaml_file):
        sys.exit(f"❌ [Error] 找不到配置文件: {yaml_file}")
    with open(yaml_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
        return data.get('config_path')

def auto_find_dataset_root(base_path):
    target_file = 'instances_train2017.json'
    base_path = os.path.abspath(base_path)
    if os.path.exists(os.path.join(base_path, 'annotations', target_file)):
        return base_path
    for root, _, files in os.walk(base_path):
        if target_file in files:
            return os.path.dirname(root)
    return base_path

# ==============================================================================
# 3. 日志 Hook
# ==============================================================================
@HOOKS.register_module()
class SdaaLogHook(Hook):
    def __init__(self, log_file='./train.log', total_steps=100):
        self.log_file = os.path.abspath(log_file)
        self.total_steps = total_steps
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, 'w') as f:
            f.write(f"==== Training Start (Total Steps: {total_steps}) ====\n")

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        current_step = runner.iter + 1
        loss = outputs['loss'].item() if 'loss' in outputs else 0.0
        # 兼容不同 optimizer 结构
        if isinstance(runner.optim_wrapper.get_lr(), dict):
            lr = runner.optim_wrapper.get_lr()['lr'][0]
        else:
            lr = runner.optim_wrapper.get_lr()[0]
            
        log_str = f"Iter[{current_step}] step_train_loss {loss:.4f} lr {lr:.2e}"
        print(log_str, flush=True)
        with open(self.log_file, 'a') as f:
            f.write(log_str + "\n")

# ==============================================================================
# 4. 核心配置构建函数
# ==============================================================================
def build_auto_config(config_rel_path, args, final_work_dir):
    cfg = Config.fromfile(os.path.join(MMDET_ROOT, config_rel_path))

    # -------------------------------------------------------------------------
    # A. 确定目标尺寸 (Target Scale)
    # -------------------------------------------------------------------------
    target_scale = (640, 640) 
    if cfg.get('test_pipeline'):
        for t in cfg.test_pipeline:
            if t['type'] == 'Resize':
                target_scale = t.get('scale', target_scale)
                break
    print(f"[AutoConfig] Standard target scale detected: {target_scale}")

    # -------------------------------------------------------------------------
    # B. Pipeline 清洗函数 (严格模式)
    # -------------------------------------------------------------------------
    def clean_pipeline(pipeline):
        if not pipeline:
            return []
        
        # 1. 明确的黑名单
        forbidden_transforms = {
            'Mosaic', 'MixUp', 'RandomFlip', 'RandomAffine', 'RandomCrop',
            'PhotoMetricDistortion', 'YOLOXHSVRandomAug', 'MinIoURandomCrop',
            'Expand', 'Corrupt', 'Albu', 'RandomShift', 'RandomCenterCropPad',
            'AutoAugment', 'RandAugment'
        }

        new_pipeline = []
        for trans in pipeline:
            t_type = trans['type']

            if t_type in forbidden_transforms:
                continue

            # 处理 Resize：强制固定
            if t_type == 'Resize' or t_type == 'RandomResize':
                trans_copy = trans.copy()
                trans_copy['type'] = 'Resize'
                
                original_scale = trans.get('scale', target_scale)
                if isinstance(original_scale, list):
                    trans_copy['scale'] = original_scale[0]
                else:
                    trans_copy['scale'] = original_scale

                # 移除随机参数
                for key in ['ratio_range', 'scale_factor', 'keep_scale_factor', 'random_scale_range']:
                    if key in trans_copy:
                        del trans_copy[key]
                
                trans_copy['keep_ratio'] = True
                new_pipeline.append(trans_copy)
                continue

            new_pipeline.append(trans)
        
        return new_pipeline

    # -------------------------------------------------------------------------
    # C. Dataset 处理
    # -------------------------------------------------------------------------
    def get_deterministic_dataset_cfg(original_dataset_cfg, is_train):
        dataset_type = original_dataset_cfg['type']
        
        if dataset_type == 'MultiImageMixDataset':
            print(f"  [Info] Unwrapping MultiImageMixDataset for {'Train' if is_train else 'Val'}")
            inner_dataset = original_dataset_cfg['dataset']
            inner_pipeline = inner_dataset.get('pipeline', [])
            outer_pipeline = original_dataset_cfg.get('pipeline', [])
            combined_pipeline = inner_pipeline + outer_pipeline
            new_dataset = inner_dataset.copy()
            new_dataset['pipeline'] = clean_pipeline(combined_pipeline)
            
        else:
            new_dataset = original_dataset_cfg.copy()
            if 'pipeline' in new_dataset:
                new_dataset['pipeline'] = clean_pipeline(new_dataset['pipeline'])
            elif is_train and cfg.get('train_pipeline'):
                new_dataset['pipeline'] = clean_pipeline(cfg.train_pipeline)
            elif not is_train and cfg.get('test_pipeline'):
                new_dataset['pipeline'] = clean_pipeline(cfg.test_pipeline)

        new_dataset['data_root'] = args.datapath
        new_dataset['ann_file'] = 'annotations/instances_train2017.json'
        new_dataset['data_prefix'] = dict(img='train2017/')
        new_dataset['metainfo'] = dict(classes=('dog', 'cat'))
        new_dataset['filter_cfg'] = dict(filter_empty_gt=is_train, min_size=32) if is_train else dict(filter_empty_gt=False)
        
        return new_dataset

    # -------------------------------------------------------------------------
    # D. 应用 Dataset 和 DataLoader
    # -------------------------------------------------------------------------
    def get_dataloader(is_train):
        raw_cfg = cfg.train_dataloader if is_train else cfg.val_dataloader
        raw_dataset = raw_cfg.dataset
        
        return dict(
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            sampler=dict(type='DefaultSampler', shuffle=is_train),
            dataset=get_deterministic_dataset_cfg(raw_dataset, is_train)
        )

    cfg.train_dataloader = get_dataloader(True)
    cfg.val_dataloader = get_dataloader(False)
    cfg.test_dataloader = cfg.val_dataloader

    # -------------------------------------------------------------------------
    # E. 模型与其他配置
    # -------------------------------------------------------------------------
    if hasattr(cfg.model, 'roi_head') and hasattr(cfg.model.roi_head, 'bbox_head'):
        cfg.model.roi_head.bbox_head.num_classes = 2
    if hasattr(cfg.model, 'bbox_head'):
        cfg.model.bbox_head.num_classes = 2
    
    if args.load_weights:
        print(f"[Init] Loading weights from: {args.load_weights}")
        cfg.model.backbone.init_cfg = None
        cfg.load_from = args.load_weights
    else:
        cfg.model.backbone.init_cfg = None
        cfg.load_from = None

    ann_file_full = os.path.join(args.datapath, 'annotations/instances_train2017.json')
    
    if args.no_validate:
        cfg.train_cfg = dict(type='IterBasedTrainLoop', max_iters=args.steps)
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None
        cfg.test_cfg = None
        cfg.test_dataloader = None
        cfg.test_evaluator = None
    else:
        cfg.train_cfg = dict(type='IterBasedTrainLoop', max_iters=args.steps, val_interval=args.steps)
        if not cfg.get('val_cfg'): cfg.val_cfg = dict(type='ValLoop')
        if not cfg.get('test_cfg'): cfg.test_cfg = dict(type='TestLoop')
        
        cfg.val_evaluator = dict(type='CocoMetric', metric='bbox', ann_file=ann_file_full)
        cfg.test_evaluator = cfg.val_evaluator

    cfg.work_dir = final_work_dir
    cfg.experiment_name = '.'
    cfg.log_level = 'WARNING'
    cfg.default_hooks.logger = dict(type='LoggerHook', interval=1)
    
    target_log_file = args.log_file if args.log_file else os.path.join(final_work_dir, 'train_loss.txt')
    if args.log_file: os.makedirs(os.path.dirname(target_log_file), exist_ok=True)

    cfg.custom_hooks = [dict(type='SdaaLogHook', total_steps=args.steps, log_file=target_log_file)]
    cfg.default_hooks.checkpoint = dict(type='CheckpointHook', interval=args.steps, by_epoch=False, max_keep_ckpts=1)

    # -------------------------------------------------------------------------
    # F. 训练策略确定性调整 (🔥 AMP 默认开启)
    # -------------------------------------------------------------------------
    if args.no_amp:
        print("[Config] AMP disabled by user (--no-amp). Using FP32.")
        # 如果原始是 Amp，退回 OptimWrapper
        if cfg.optim_wrapper.type == 'AmpOptimWrapper':
            cfg.optim_wrapper.type = 'OptimWrapper'
            if 'loss_scale' in cfg.optim_wrapper:
                del cfg.optim_wrapper['loss_scale']
    else:
        print("[Config] AMP enabled by default.")
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    if args.auto_scale_lr:
        cfg.auto_scale_lr = dict(enable=True, base_batch_size=args.batch_size)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    return cfg

# ==============================================================================
# 5. 主程序
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--datapath', type=str, default='./data')
    parser.add_argument('--load-weights', type=str, default=False)
    parser.add_argument('--name', type=str, default='train')
    parser.add_argument('--seed', type=int, default=42)
    
    # 🔥 修改：--no-amp 用于关闭，默认是开启的
    parser.add_argument('--no-amp', action='store_true', help='Disable AMP (Force FP32)')
    
    parser.add_argument('--no-validate', action='store_true')
    parser.add_argument('--auto-scale-lr', action='store_true')
    parser.add_argument('--no-pin-memory', action='store_true')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--log-file', type=str, default=None)

    return parser.parse_args()

def to_camel_case(snake_str):
    components = snake_str.split('_')
    return ''.join(x.title() for x in components)

def main():
    args = parse_args()
    set_deterministic_context(args.seed)
    
    register_all_modules(init_default_scope=False)
    if not torch.cuda.is_available():
        sys.exit("❌ [Error] CUDA not available")

    config_rel_path = load_model_yaml(args.model)
    args.datapath = auto_find_dataset_root(args.datapath)

    final_work_dir = os.path.join(os.getcwd(), 'result', args.name.capitalize())
    os.makedirs(final_work_dir, exist_ok=True)
    
    if not args.log_file:
        model_camel = to_camel_case(args.model)
        args.log_file = os.path.join(final_work_dir, f"{model_camel}{args.name.capitalize()}.log")

    cfg = build_auto_config(config_rel_path, args, final_work_dir)

    runner = Runner.from_cfg(cfg)

    if not args.load_weights:
        save_path = './random_init_weights.pth'
        print(f"[Init] Saving random initialization to: {save_path}")
        torch.save(runner.model.state_dict(), save_path)
    
    runner.train()

if __name__ == '__main__':
    main()