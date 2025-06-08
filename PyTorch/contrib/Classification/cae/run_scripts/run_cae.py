# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
from copy import deepcopy
from torch_sdaa.utils import cuda_migrate  # 使用torch_sdaa自动迁移方法
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.registry import RUNNERS, HOOKS
from mmengine.runner import Runner
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
from mmengine.hooks import Hook
from datetime import datetime

# 注册自定义钩子
@HOOKS.register_module()
class CustomLogHook(Hook):
    """日志记录钩子，在每次训练迭代后记录指标"""
    priority = 70  # 在 LogProcessor（60）之后但在 CheckpointHook（80）之前执行
    
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.log_file = None
        self.log_file_path = None
    
    def before_run(self, runner):
        # 训练开始时设置起始时间
        self.start_time = time.time()
        
        # 确定日志文件路径
        logs_dir = osp.join(runner.work_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        print("===========================",logs_dir)
       # logs_dir="/data/bigc-data/zym/mmpretrain/run_scripts" ##run scripts要求生成的sdaa.log要在run_scripts文件夹下，这里手动指定run_scripts目录
        self.log_file_path = osp.join(logs_dir, 'sdaa.log')
        
        # 打开日志文件
        try:
            self.log_file = open(self.log_file_path, 'a')
            runner.logger.info(f"Logging to file: {self.log_file_path}")
        except Exception as e:
            runner.logger.error(f"Failed to open log file: {e}")
    
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        # 只由 rank 0 记录日志
        if runner.world_size > 1 and runner.rank != 0:
            return
        
        # 获取当前指标值
        metrics = {}
        
        if 'train/loss' in runner.message_hub.log_scalars:
            loss_buffer = runner.message_hub.get_scalar('train/loss')
            metrics['train.loss'] = loss_buffer.current()
        
        if 'train/ips' in runner.message_hub.log_scalars:
            ips_buffer = runner.message_hub.get_scalar('train/ips')
            metrics['train.ips'] = ips_buffer.current()
        
        # 计算训练总时间
        if self.start_time is not None:
            metrics['train.total_time'] = time.time() - self.start_time
        
        # 记录所有收集到的指标
        if metrics:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            iter_info = f"Epoch: {runner.epoch} Iteration: {runner.iter} rank: {runner.rank}"
            log_line = f"{timestamp} - {iter_info}"
            
            for k, v in metrics.items():
                if k == 'train.ips':
                    log_line += f" {k} : {v:.3f} imgs/s"
                else:
                    log_line += f" {k} : {v}"
            
            # 构造完整的日志行
            full_log_line = f"TCAPPDLL {log_line}"
            
            # 输出到控制台和文件
            print(full_log_line)
            if self.log_file:
                self.log_file.write(full_log_line + "\n")
                self.log_file.flush()  # 确保立即写入
    
    def after_run(self, runner):
        # 关闭日志文件
        if self.log_file:
            self.log_file.close()
            self.log_file = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='whether to auto scale the learning rate according to the '
        'actual batch size and the original batch size.')
    parser.add_argument(
        '--no-pin-memory',
        action='store_true',
        help='whether to disable the pin_memory option in dataloaders.')
    parser.add_argument(
        '--no-persistent-workers',
        action='store_true',
        help='whether to disable the persistent_workers option in dataloaders.'
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local_rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    # resume training
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # enable auto scale learning rate
    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    # set dataloader args
    default_dataloader_cfg = ConfigDict(
        pin_memory=True,
        persistent_workers=True,
        collate_fn=dict(type='default_collate'),
    )
    if digit_version(TORCH_VERSION) < digit_version('1.8.0'):
        default_dataloader_cfg.persistent_workers = False

    def set_default_dataloader_cfg(cfg, field):
        if cfg.get(field, None) is None:
            return
        dataloader_cfg = deepcopy(default_dataloader_cfg)
        dataloader_cfg.update(cfg[field])
        cfg[field] = dataloader_cfg
        if args.no_pin_memory:
            cfg[field]['pin_memory'] = False
        if args.no_persistent_workers:
            cfg[field]['persistent_workers'] = False

    set_default_dataloader_cfg(cfg, 'train_dataloader')
    set_default_dataloader_cfg(cfg, 'val_dataloader')
    set_default_dataloader_cfg(cfg, 'test_dataloader')

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


def main():
    args = parse_args()
    
    # 加载配置文件
    cfg = Config.fromfile(args.config)
    
    # 合并命令行参数到配置中
    cfg = merge_args(cfg, args)
    
    # 添加自定义日志钩子
    cfg.custom_hooks = [dict(type='CustomLogHook')]

    # 使用合并后的 cfg.work_dir
    logs_dir = osp.join(cfg.work_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)  # 创建日志目录
    
    # 初始化 Logger（使用 cfg.work_dir）
    json_logger = Logger(
        [
            StdOutBackend(Verbosity.DEFAULT),
            JSONStreamBackend(Verbosity.VERBOSE, osp.join(logs_dir, 'custom_sdaa.log')),
        ]
    )
    
    # 定义元数据
    json_logger.metadata("train.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
    json_logger.metadata("train.ips", {"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})
    json_logger.metadata("train.total_time", {"unit": "s", "format": ":.3f", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})

    # 构建并启动 runner
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.train()


if __name__ == '__main__':
    main()