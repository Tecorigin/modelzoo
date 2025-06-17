# # BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# # reserved.
# # Redistribution and use in source and binary forms, with or without
# # modification, are permitted provided that the following conditions are met:
# # Redistributions of source code must retain the above copyright notice,
# # this list of conditions and the following disclaimer.
# # Redistributions in binary form must reproduce the above copyright notice,
# # this list of conditions and the following disclaimer in the documentation
# # and/or other materials provided with the distribution.
# # Neither the name of the copyright holder nor the names of its contributors
# # may be used to endorse or promote products derived from this software
# # without specific prior written permission.
# #
# # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# # ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# # LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# # CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# # SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# # INTERRUPTION)
# # HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# # STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# # WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# # OF SUCH DAMAGE.
# import argparse
# import os
# import os.path as osp
# import time
# import mmpretrain.visualization
# from copy import deepcopy
# from torch_sdaa.utils import cuda_migrate  # 使用torch_sdaa自动迁移方法
# from mmengine.config import Config, ConfigDict, DictAction
# from mmengine.registry import RUNNERS, HOOKS
# from mmengine.runner import Runner
# from mmengine.utils import digit_version
# from mmengine.utils.dl_utils import TORCH_VERSION
# from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
# from mmengine.hooks import Hook
# from datetime import datetime

# BSD 3-Clause License
# Copyright (c) 2023, Tecorigin Co., Ltd. All rights reserved.
# BSD 3-Clause License
# Copyright (c) 2023, Tecorigin Co., Ltd. All rights reserved.
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
        self.last_step_time = None
        self.batch_size = None
    
    def before_run(self, runner):
        self.start_time = time.time()
        self.last_step_time = time.time()
        
        # 获取批量大小
        self.batch_size = runner.train_dataloader.batch_size
        
        # 确定日志文件路径
        logs_dir = osp.join(runner.work_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        self.log_file_path = osp.join(logs_dir, 'sdaa.log')
        
        try:
            self.log_file = open(self.log_file_path, 'a')
            runner.logger.info(f"Logging to file: {self.log_file_path}")
        except Exception as e:
            runner.logger.error(f"Failed to open log file: {e}")
    
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        current_time = time.time()
        
        # 获取当前指标值
        metrics = {}
        
        if 'train/loss' in runner.message_hub.log_scalars:
            loss_buffer = runner.message_hub.get_scalar('train/loss')
            metrics['train.loss'] = loss_buffer.current()
        
        # 计算 images per second (ips)
        step_time = current_time - self.last_step_time
        if step_time > 0 and self.batch_size is not None:
            ips = self.batch_size / step_time
            metrics['train.ips'] = ips
        
        # 计算训练总时间
        if self.start_time is not None:
            metrics['train.total_time'] = current_time - self.start_time
        
        # 更新上一步时间
        self.last_step_time = current_time
        
        # 记录所有收集到的指标
        if metrics:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            iter_info = f"Epoch: {runner.epoch} Iteration: {runner.iter}  rank : {runner.rank}"
            log_parts = [
                timestamp,
                "-",
                iter_info
            ]
            
            if 'train.loss' in metrics:
                log_parts.append(f" train.loss : {metrics['train.loss']:.10f}")
            if 'train.ips' in metrics:
                log_parts.append(f" train.ips : {metrics['train.ips']:.10f} imgs/s")
            if 'train.total_time' in metrics:
                log_parts.append(f" train.total_time : {metrics['train.total_time']:.10f}")
            
            # 构造完整的日志行
            full_log_line = f"TCAPPDLL {' '.join(log_parts)}"
            
            # 输出到控制台和文件
            print(full_log_line)
            if self.log_file:
                self.log_file.write(full_log_line + "\n")
                self.log_file.flush()
    
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
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
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