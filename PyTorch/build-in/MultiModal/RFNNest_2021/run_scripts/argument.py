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

import torch
import argparse
from argparse import ArgumentParser,ArgumentTypeError
import sys
import warnings

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )
    
def parse_args():
    """Command-line argument parser for training."""

    parser = ArgumentParser(description="modelzoo")
    parser.add_argument("--model_name", required=True, default="RFNNest", type=str, help="name of the model")


    parser.add_argument('--RFN',
                        default=False, type=bool, help='判断训练阶段')
    parser.add_argument('--image_path_autoencoder',
                        default=r'/mnt_qne00/dataset/coco/train2017/', type=str, help='数据集路径')
    parser.add_argument('--image_path_rfn',
                        default=r'../dataset/KAIST', type=str, help='数据集路径')
    parser.add_argument('--gray',
                        default=True, type=bool, help='是否使用灰度模式')
    parser.add_argument('--train_num',
                        default=1600, type=int, help='用于训练的图像数量')
    # 训练相关参数
    parser.add_argument('--deepsupervision', default=False, type=bool, help='是否深层监督多输出')
    parser.add_argument('--resume_nestfuse',
                        default=None, type=str, help='导入已训练好的模型路径')
    parser.add_argument('--resume_rfn',
                        default=None, type=str, help='导入已训练好的模型路径')
    parser.add_argument('--device', type=str, default='sdaa', help='训练设备')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size, default=4')
    parser.add_argument('--num_workers', type=int, default=0, help='载入数据集所调用的cpu线程数')
    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs to train for, default=10')
    parser.add_argument('--lr', type=float, default=1e-4, help='select the learning rate, default=1e-2')
    # 打印输出
    parser.add_argument('--output', action='store_true', default=True, help="shows output")
 
    return parser.parse_args()

def check_argument(args):
    # check model_name
    assert args.model_name in ["RFNNest"], "model_name should be RFNNest"
    return args
    

if __name__ == "__main__":
    sys.exit(0)