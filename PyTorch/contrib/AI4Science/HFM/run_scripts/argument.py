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

def parse_arguments():
    parser = argparse.ArgumentParser(description='CFD HFM Training')
    parser.add_argument("--batch_size", default=40000, type=int)
    parser.add_argument("--T_data", required=True, type=int)
    parser.add_argument("--N_data", required=True, help='Total number of points', type=int)
    parser.add_argument("--work_size", default=2, type=int)
    parser.add_argument("--path", default="/home/hpc/cfd/Datasets_HFM/gen_data_predict_t1.npy", type=str)
    parser.add_argument("--total_epoch", default=50,type=int)
    parser.add_argument("--is_eqns", default=True, help='The flag of auto generate dataset.', type=bool)
    parser.add_argument("--is_other_loss", default=False, help='The flag of using other loss.', type=bool)
    parser.add_argument("--model_path", default=None, help='Loading weights from this path.')
    parser.add_argument("--alpha", default=0.7, help='The weight of Ohter loss ', type=float)
    parser.add_argument("--rey", default=100, type=int)
    parser.add_argument("--pec", default=100, type=int)
    parser.add_argument("--layers", default=10, type=int)
    parser.add_argument("--width", default=200, type=int)
    parser.add_argument("--activation", default='swish', type=str)
    parser.add_argument("--normalization", default=None, type=str)
    parser.add_argument("--model_save_path", default="weights/Cylinder3D.pth", type=str)
    parser.add_argument("--multi_machine", action='store_true')
    parser.add_argument("--master_addr",default="127.0.0.1")
    parser.add_argument("--master_port", default='29500')
    parser.add_argument("--node_rank", default=0, type=int)
    parser.add_argument("--local_size",type=int, default=2)
    parser.add_argument('--device', required=True, default='cpu', type=str,
                        help='which device to use. cpu, cuda, sdaa optional, cpu default')
    parser.add_argument("--ddp", action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    sys.exit(0)
