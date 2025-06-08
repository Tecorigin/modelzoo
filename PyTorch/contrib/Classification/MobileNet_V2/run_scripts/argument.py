# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.
import argparse
import os
from mmengine.config import DictAction

def parse_args():
    parser = argparse.ArgumentParser(description='Unified training interface for SAE models')

    # 基本训练参数
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--work-dir', type=str, default=None, help='Directory to save logs and checkpoints')
    parser.add_argument('--model-name', type=str, default='ResNext', help='Name of the model')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=300, help='Total number of training epochs')
    parser.add_argument('--dataset-root', type=str, default='/data/teco-data/imagenet/', help='Root path to dataset')

    # Mixed precision 和 恢复训练相关
    parser.add_argument('--amp', action='store_true', help='Enable automatic mixed precision')
    parser.add_argument('--resume', nargs='?', type=str, const='auto', help='Checkpoint path or auto to resume latest')
    parser.add_argument('--no-validate', action='store_true', help='Disable validation during training')
    parser.add_argument('--auto-scale-lr', action='store_true', help='Auto-scale learning rate based on batch size')

    # dataloader 优化选项
    parser.add_argument('--no-pin-memory', action='store_true', help='Disable pin_memory in dataloaders')
    parser.add_argument('--no-persistent-workers', action='store_true', help='Disable persistent workers in dataloaders')

    # mmengine 相关
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the config file, in key=value format')

    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='Job launcher type')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args