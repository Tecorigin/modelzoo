#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import argparse
import random
import yaml
import numpy as np
import torch
import shutil

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmdet.utils import register_all_modules

# ==============================================================================
# 1. 基础路径配置
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MMDET = os.path.join(CURRENT_DIR, '../mmdetection')

if os.path.exists(LOCAL_MMDET):
    print(f"[Mode] 检测到本地 mmdetection，进入【独立/打包模式】")
    MMDET_ROOT = LOCAL_MMDET
else:
    print(f"[Mode] 未检测到本地 mmdetection，进入【开发模式】")
    MMDET_ROOT = '/data/application/wangwl/Detection/mmdetection'

# ==============================================================================
# 2. 辅助工具
# ==============================================================================
def load_model_yaml(model_name):
    yaml_file = os.path.join(CURRENT_DIR, f"{model_name}.yml")
    if not os.path.exists(yaml_file):
        print(f"❌ [Error] 找不到配置文件: {yaml_file}")
        sys.exit(1)
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

def check_device_availability():
    if not torch.cuda.is_available():
        print("❌ [Error] 未检测到 CUDA 设备")
        sys.exit(1)

# ==============================================================================
# 3. 环境设置
# ==============================================================================
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "12345"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Setup] Global Seed: {seed}")

# ==============================================================================
# 4. 日志 Hook
# ==============================================================================
@HOOKS.register_module()
class SdaaLogHook(Hook):
    def __init__(self, log_file='./train.log', total_steps=100):
        self.log_file = os.path.abspath(log_file)
        self.total_steps = total_steps
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, 'w') as f:
            f.write(f"==== Training Start (Total Steps: {total_steps}) ====\n")
        print(f"==== Epoch 0 (IterBased Mode) | Log: {self.log_file} ====", flush=True)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        current_step = runner.iter + 1
        loss = outputs['loss'].item() if 'loss' in outputs else 0.0
        lr = runner.optim_wrapper.get_lr()['lr'][0]
        log_str = f"Iter[{current_step}] step_train_loss {loss:.4f} lr {lr:.2e}"
        print(log_str, flush=True)
        with open(self.log_file, 'a') as f:
            f.write(log_str + "\n")

# ==============================================================================
# 5. Config 构建
# ==============================================================================
def build_auto_config(config_rel_path, args, final_work_dir):
    cfg = Config.fromfile(os.path.join(MMDET_ROOT, config_rel_path))

    # 类别数修改
    if hasattr(cfg.model, 'roi_head'):
        cfg.model.roi_head.bbox_head.num_classes = 2
    if hasattr(cfg.model, 'bbox_head'):
        cfg.model.bbox_head.num_classes = 2
    
    if args.load_weights:
        print(f"[Init] Loading weights from: {args.load_weights}")
        cfg.model.backbone.init_cfg = None
        cfg.load_from = args.load_weights
    else:
        # cfg.model.backbone.init_cfg = None
        cfg.load_from = None

    metainfo = dict(classes=('dog', 'cat'))

    # ============================================================
    # 关键修复 1: 获取原始配置中的 pipeline
    # ============================================================
    train_pipeline = cfg.train_pipeline
    test_pipeline = cfg.test_pipeline

    # ============================================================
    # 关键修复 2: 函数接收 pipeline 参数，并传给 dataset
    # ============================================================
    def get_dataloader_cfg(shuffle=False, pipeline=None):
        # ... (此处代码保持不变) ...
        proposal_file_path = os.path.join(args.datapath, 'proposals/proposals_train2017_final.pkl')
        
        final_proposal_file = None
        if os.path.exists(proposal_file_path):
            print(f"[Dataset] ✅ 成功定位 Proposal 文件: {proposal_file_path}")
            final_proposal_file = proposal_file_path
        else:
            if 'fast_rcnn' in args.model or 'fast-rcnn' in args.model:
                print(f"[Dataset] ❌ 严重错误: Fast R-CNN 需要 proposal 文件 but not found")
            else:
                print(f"[Dataset] ⚠️ 未找到 Proposal 文件 (非 Fast R-CNN 可忽略)")

        return dict(
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=not args.no_pin_memory,
            persistent_workers=False,
            sampler=dict(type='DefaultSampler', shuffle=shuffle),
            dataset=dict(
                type='CocoDataset',
                data_root=args.datapath,
                metainfo=metainfo,
                ann_file='annotations/instances_train2017.json',
                data_prefix=dict(img='train2017/'),
                proposal_file=final_proposal_file,
                pipeline=pipeline
            )
        )

    # 传入对应的 pipeline
    cfg.train_dataloader = get_dataloader_cfg(False, train_pipeline)
    cfg.val_dataloader = get_dataloader_cfg(False, test_pipeline)
    cfg.test_dataloader = cfg.val_dataloader
    
    ann_file_rel = 'annotations/instances_train2017.json'
    ann_file_full = os.path.join(args.datapath, ann_file_rel)

    # ---------------- no-validate ----------------
    if args.no_validate:
        cfg.train_cfg = dict(type='IterBasedTrainLoop', max_iters=args.steps)
        cfg.val_dataloader = None
        cfg.val_evaluator = None
        cfg.test_dataloader = None
        cfg.test_evaluator = None
        cfg.val_cfg = None
        cfg.test_cfg = None
        print("[Config] Validation disabled")
    else:
        cfg.train_cfg = dict(type='IterBasedTrainLoop', max_iters=args.steps, val_interval=args.steps)
        cfg.val_evaluator = dict(
            type='CocoMetric', 
            metric='bbox', 
            ann_file=ann_file_full
        )
        cfg.test_evaluator = cfg.val_evaluator

    cfg.work_dir = final_work_dir
    cfg.experiment_name = '.'
    cfg.log_level = 'WARNING'
    cfg.default_hooks.logger = dict(type='LoggerHook', interval=1)

    if hasattr(args, 'log_file') and args.log_file:
        target_log_file = args.log_file
        os.makedirs(os.path.dirname(target_log_file), exist_ok=True)
    else:
        target_log_file = os.path.join(final_work_dir, 'train_loss.txt')
    
    cfg.custom_hooks = [
        dict(
            type='SdaaLogHook', 
            total_steps=args.steps, 
            log_file=target_log_file
        )
    ]

    cfg.default_hooks.checkpoint = dict(
        type='CheckpointHook',
        interval=args.steps,
        by_epoch=False,
        max_keep_ckpts=1
    )

    # ---------------- AMP ----------------
    if args.amp:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'
        # 确保 AMP 模式下梯度裁剪依然生效
        if cfg.optim_wrapper.get('clip_grad') is None:
             cfg.optim_wrapper.clip_grad = dict(max_norm=35, norm_type=2)

    # ---------------- auto-scale-lr ----------------
    if args.auto_scale_lr:
        cfg.auto_scale_lr = dict(enable=True, base_batch_size=args.batch_size)
        print("[Config] Auto scale LR enabled")

    # ---------------- cfg-options ----------------
    if args.cfg_options:
        print(f"[Config] Apply cfg-options: {args.cfg_options}")
        cfg.merge_from_dict(args.cfg_options)

    return cfg

# ==============================================================================
# 6. 参数解析
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--val-num', type=int, default=50)
    parser.add_argument('--datapath', type=str, default='./data')
    parser.add_argument('--load-weights', type=str, default=False)
    parser.add_argument('--name', type=str, default='train')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--amp', action='store_true')

    # 🔥 新增参数
    parser.add_argument('--no-validate', action='store_true')
    parser.add_argument('--auto-scale-lr', action='store_true')
    parser.add_argument('--no-pin-memory', action='store_true')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)

    return parser.parse_args()

# ==============================================================================
# 7. Main
# ==============================================================================
def to_camel_case(snake_str):
    """faster_rcnn -> FasterRcnn"""
    components = snake_str.split('_')
    return ''.join(x.title() for x in components)

def main():
    args = parse_args()
    set_global_seed(args.seed)
    register_all_modules(init_default_scope=False)
    check_device_availability()

    config_rel_path = load_model_yaml(args.model)
    args.datapath = auto_find_dataset_root(args.datapath)

    work_root = os.getcwd()
    final_work_dir = os.path.join(work_root, 'result', args.name.capitalize())
    os.makedirs(final_work_dir, exist_ok=True)

    model_camel = to_camel_case(args.model)
    run_name_cap = args.name.capitalize()
    log_name = f"{model_camel}{run_name_cap}.log"

    args.log_file = os.path.join(final_work_dir, log_name)

    cfg = build_auto_config(config_rel_path, args, final_work_dir)

    # if args.load_weights:
    #     cfg.load_from = args.load_weights
    #     print(f"[Load] weights from {args.load_weights}")
    if args.load_weights:
        print(f"[Init] Load backbone pretrained: {args.load_weights}")
        cfg.load_from = None
        cfg.model.backbone.init_cfg = dict(
            type='Pretrained',
            checkpoint=args.load_weights
        )

    runner = Runner.from_cfg(cfg)

    if not args.load_weights:
        save_path = './random_init_weights.pth'  # 当前目录
        print(f"[Init] Saving random initialization to: {save_path}")
        torch.save(runner.model.state_dict(), save_path)
        print("[Init] Done. Exiting...")

    runner.train()

if __name__ == '__main__':
    main()