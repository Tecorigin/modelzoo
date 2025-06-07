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

import argparse
import os
import os.path as osp
import time
from copy import deepcopy
from torch_sdaa.utils import cuda_migrate  # 使用torch_sdaa自动迁移方法
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity


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
        help='whether to disable the persistent_workers option in dataloaders.')
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

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    # 为了适配太初加速卡float16卷积优化，建议使用AMP并开启GradScaler防止数值溢出
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

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

    # 创建日志目录
    logs_dir = osp.join(cfg.work_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # 初始化 TCAP DLLogger
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

    # 构建 runner 并启动训练
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.train()


if __name__ == '__main__':
    main()
