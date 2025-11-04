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
                        default='resnet50', type=str, help='name of the model')
    parser.add_argument('--epoch', required=False, default=1,
                        type=int, help='number of total epochs to run')
    parser.add_argument('--step', required=False, default=-1,
                        type=int, help='run only N iterations')
    parser.add_argument('--batch_size','--bs' ,required=True, default=64,
                        type=int, help='mini-batch size (default: 64) per device')
    parser.add_argument('--dataset_path', required=True,
                        default=None, type=str, help='path to dataset')
    parser.add_argument('--nproc_per_node', required=False, default=1, type=int, help="The number of processes to launch on each node, "
                        "for GPU training, this is recommended to be set "
                        "to the number of GPUs in your system so that "
                        "each process can be bound to a single GPU."
                        )
    parser.add_argument('--nnode', required=False, default=1, type=int,
                        help="The number of nodes to use for distributed " "training")
    parser.add_argument('--node_rank', required=False, default=0, type=int,
                        help="The rank of the node for multi-node distributed " "training")
    parser.add_argument('--master_addr', required=False, default='127.0.0.1', type=str,
                        help="Master node (rank 0)'s address, should be either "
                        "the IP address or the hostname of node 0, for "
                        "single node multi-proc training, the "
                        "--master_addr can simply be 127.0.0.1")
    parser.add_argument('--master_port', required=False, default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                        "be used for communciation during distributed "
                        "training")
    parser.add_argument('--precision_align', default=False,
                        type=str2bool, help='do precision align')
    parser.add_argument('--precision_align_cuda_path', default=None,
                        type=str, help='path to cuda data of the precision_align')
    parser.add_argument('--precision_align_log_path',  default=None,
                        type=str, help='path to output of the precision_align')
    parser.add_argument('--lr', "--learning-rate", required=True,
                        default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='number of data loading workers ')
    parser.add_argument('--device', required=True, default='cpu', type=str,
                        help='which device to use. cpu, cuda, sdaa optional, cpu default')
    parser.add_argument("--profiler", default=False,
                        type=str2bool, help="run profiler")
    parser.add_argument("--autocast", default=False,
                        type=str2bool, help="open autocast for amp")
    parser.add_argument("--grad_scale", default=False,
                        type=str2bool, help="open grads_cale for amp")
    parser.add_argument("--scale_loss", default=1024.0,
                        type=float, help="scale loss for amp")
    parser.add_argument("--amp_blacklist", default="",
                        type=str, help="custom black list for amp")
    parser.add_argument("--amp_whitelist", default="",
                        type=str, help="custom white list for amp")
    parser.add_argument("--profiler_path", default='./profiler_log',
                        type=str, help="profiler save path")
    parser.add_argument("--early_stop", default=-1,
                        type=int, help="early stop")


    return parser.parse_args()


def check_argument(args):
    if args.step>0 and args.epoch>1:
        args.epoch=1
        warnings.warn('argument step and epoch is conflict, when step is used, epoch will be set to 1')
    if args.nproc_per_node >1 and args.nnode >1 and args.precision_align:
        warnings.warn('precision_align only for single device')
        sys.exit(0)
    if args.precision_align and not (args.precision_align_cuda_path and args.precision_align_log_path):
        warnings.warn('please pass the full argument for precision_align')
        sys.exit(0)
    if args.precision_align and args.autocast:
        warnings.warn('precision_align are not support with amp training')
        sys.exit(0)
    return args
    

if __name__ == '__main__':
    sys.exit(0)
