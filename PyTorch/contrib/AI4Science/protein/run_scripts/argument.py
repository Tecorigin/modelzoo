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
    parser = argparse.ArgumentParser(description='PROTEIN Training')
    parser.add_argument("--batch_size", required=True, default=64, type=int)
    parser.add_argument("--train_size", required=True, default=64, type=int)
    parser.add_argument("--stop_rounds", required=False, default=100, type=int)
    parser.add_argument("--save_rounds", required=True, default=10, type=int)
    parser.add_argument("--epoch", required=True, default=10,type=int)
    parser.add_argument("--augmentation", required=False, default=0.1, type=float)
    parser.add_argument("--k_fold", required=True, default=3,type=int)
    parser.add_argument("--save_path", required=True, default="model/FinalModel", type=str)
    parser.add_argument("--ddp", required=False, action='store_true')
    parser.add_argument('--device', required=True, default='cpu', type=str,
                        help='which device to use. cpu, cuda, sdaa optional, cpu default')
    parser.add_argument('--nproc_per_node', required=False, default=1, type=int)
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument("--data_seq", type=str, default='./data_seq_train.txt', help='Path to the dataset.')
    parser.add_argument("--data_sec", type=str, default='./data_sec_train.txt', help='Path to the dataset.')


    return parser.parse_args()




if __name__ == '__main__':
    sys.exit(0)
