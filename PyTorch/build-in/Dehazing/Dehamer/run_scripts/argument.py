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

    # New parser
    parser = ArgumentParser(description="modelzoo")
    parser.add_argument("--model_name", required=True, default="Dehamer", type=str, help="name of the model")

    # Data parameters
    parser.add_argument('-d', '--dataset-name', help='name of dataset',choices=['NH', 'dense', 'indoor','outdoor'], default='NH')
    parser.add_argument('-t', '--train_dir', help='training set path', default='./../data/train')
    parser.add_argument('-v', '--valid_dir', help='test set path', default='./../data/valid')
    parser.add_argument('--ckpt_save_path', help='checkpoint save path', default='./../ckpts')
    parser.add_argument('--ckpt_overwrite', help='overwrite model checkpoint on save', action='store_true')
    parser.add_argument('--ckpt_load_path', help='start training with a pretrained model',default=None)
    parser.add_argument('--report_interval', help='batch report interval', default=1, type=int)
    parser.add_argument('-ts', '--train_size',nargs='+', help='size of train dataset',default=[192,288], type=int)
    parser.add_argument('-vs', '--valid_size',nargs='+', help='size of valid dataset',default=[192,288], type=int)  

    # Training hyperparameters 
    parser.add_argument('-lr', '--learning_rate', help='learning rate', default=0.0001, type=float)
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
    parser.add_argument('-b', '--batch_size', help='minibatch size', default=8, type=int)
    parser.add_argument('-e', '--nb_epochs', help='number of epochs', default=100, type=int)
    parser.add_argument('-l', '--loss', help='loss function', choices=['l1', 'l2'], default='l1', type=str)
    parser.add_argument('--cuda', help='use cuda', action='store_true')
    parser.add_argument('--plot_stats', help='plot stats after every epoch', action='store_true') 
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('-c', '--crop_size', help='random crop size', default=128, type=int)#224
 
    return parser.parse_args()

def check_argument(args):
    # check model_name
    assert args.model_name in ["Dehamer"], "model_name should be Dehamer"
    return args
    

if __name__ == "__main__":
    sys.exit(0)