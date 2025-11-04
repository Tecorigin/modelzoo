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
    parser = ArgumentParser(description='modelzoo')
    parser.add_argument('--model_name', required=True,
                        default='faster_rcnn', type=str, help='name of the model')
    parser.add_argument('--nproc_per_node', required=True, default=1, type=int, help="The number of processes to launch on each node, "
                        "for GPU training, this is recommended to be set "
                        "to the number of GPUs in your system so that "
                        "each process can be bound to a single GPU."
                        )
    parser.add_argument('--batch_size','--bs' ,required=True, default=2,
                        type=int, help='mini-batch size (default: 64) per device')
    parser.add_argument('--epochs', required=True, default=1,
                        type=int, help='number of total epochs to run')
    parser.add_argument('--nnode', required=True, default=1, type=int,
                        help="The number of nodes to use for distributed " "training")
    parser.add_argument('--node_rank', required=True, default=0, type=int,
                        help="The rank of the node for multi-node distributed " "training")
    parser.add_argument('--device', default='sdaa', type=str,
                        help="Device (accelerator) to use.")
    parser.add_argument('--lr', default=None, type=float, 
                        help='learning rate, overrides lr-base if set (default: None)')
    parser.add_argument('--data-dir', default='data/cocosmall', type=str,
                        help='path to dataset (root dir)')
    parser.add_argument('--ckpt', default=None, type=str,
                        help="checkpoint.")
    return parser.parse_args()


# def check_argument(args):
#     if args.step>0 and args.epoch>1:
#         args.epoch=1
#         warnings.warn('argument step and epoch is conflict, when step is used, epoch will be set to 1')
#         sys.exit(0)
#     return args
    

if __name__ == '__main__':
    sys.exit(0)
