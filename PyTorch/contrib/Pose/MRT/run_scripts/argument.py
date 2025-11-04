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
    parser = ArgumentParser(description='modelzoo')
    parser.add_argument('--model_name', required=False,
                        default='mrt', type=str, help='name of the model')
    parser.add_argument('--epoch', required=False, default=20,
                        type=int, help='number of total epochs to run, default: 20')
    parser.add_argument('--batch_size','--bs' ,required=False, default=64,
                        type=int, help='mini-batch size per device, default: 64')
    parser.add_argument('--device', required=False, default='sdaa', type=str,
                        help="The device for this task, e.g. sdaa or cpu, default: sdaa")
    parser.add_argument('--data_path', required=False, default='./mocap', type=str,
                        help="The root path of dataset, default: ./mocap")
    parser.add_argument('--eval', default=False,
                            action='store_true', help="whether to do eval")
    parser.add_argument('--ddp', default=False, action='store_true', help="whether to use ddp")
    parser.add_argument("--autocast", default=False, action='store_true', help="open autocast for amp")
    parser.add_argument('--checkpoint', required=False, default='saved_model/19.model', type=str, help='checkpoint file path')
    return parser.parse_args()


if __name__ == '__main__':
    sys.exit(0)
