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

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from tqdm import tqdm
import random
import argparse
from argparse import ArgumentTypeError
import numpy as np
import time
import logging
import json
from datetime import datetime

import torch
import torch_sdaa
import torch.nn as nn
from torch.utils import data

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from models import FCN
from utils import validate, get_dataset, plot_train_loss, plot_val_loss, get_new_experiment_folder
import utils
from metrics import StreamSegMetrics


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


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='vaihingen',
                        choices=['vaihingen', ], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=6,
                        help="num classes (default: None)")


    parser.add_argument("--model_name", type=str, default='fcn',
                        choices=['fcn', ], help='model name')

    # Train Options
    parser.add_argument("--distributed", type=str2bool, default=False)
    parser.add_argument("--use_amp", type=str2bool, default=False)
    parser.add_argument("--default_rank",  type=int, default=0)
    parser.add_argument("--total_epochs", type=int, default=100,
                        help="epoch number (default: 30k)")
    parser.add_argument("--crop_val", type=str2bool, default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help='batch size for validation (default: 1)')
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--ckpt", default='experiments/example/best_fcn_vaihingen.pth', type=str,
                        help="restore from checkpoint")
    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument('--device', required=True, default='sdaa', type=str,
                    help='which device to use. cuda, sdaa optional, sdaa default')

    return parser


def main():
    opts = get_argparser().parse_args()
    
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    # Setup device
    if not opts.distributed:
        opts.default_rank = local_rank
        device = torch.device(opts.device if torch.sdaa.is_available() else 'cpu')
    else:
        device = torch.device(f"{opts.device}:{local_rank}" if torch.sdaa.is_available() else 'cpu')
        torch.sdaa.set_device(device)
    
    # Setup logger
    if local_rank == opts.default_rank:
        work_dir = os.path.dirname(opts.ckpt)

        # get current time for log name
        current_time = datetime.now()
        # format time information
        time_str = current_time.strftime("%Y%m%d_%H%M%S")

        logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join(work_dir, f"test_{time_str}.log")),
                        logging.StreamHandler()
                    ])
        logger = logging.getLogger("TecoSeg")
        opts_dict = vars(opts)
        formatted_opts = json.dumps(opts_dict, indent=4)
        logger.info("============arguments=============")
        logger.info(formatted_opts)
        logger.info("==================================")
        logger.info("Device: %s" % device)

    # Init processgroup, choose tccl as backend
    if opts.distributed:
        torch.distributed.init_process_group(backend="tccl", init_method="env://")

    _, _, test_dst, CLASS_NAMES = get_dataset(opts)
    
    # Setup Dataloader 
    if opts.distributed:
        # Distrubuted dataloader
        test_sampler = DistributedSampler(test_dst)
        test_loader = data.DataLoader(test_dst, batch_size=opts.val_batch_size, 
                                    sampler=test_sampler, num_workers=0)
    else:
        test_loader = data.DataLoader(
            test_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=0)
    
    if local_rank == opts.default_rank:
        logger.info("Dataset: %s, Test set: %d" %
            (opts.dataset, len(test_dst)))

    if opts.model_name == "fcn":
        model = FCN(opts.num_classes).to(device)
    else:
        raise Exception("Sorry, the model you choose is not supported now")
    # if opts.distributed:
    #     # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #     model = DDP(model)
    
    scaler = torch_sdaa.amp.GradScaler() # sdaa need this

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes, CLASS_NAMES)

    # Set up criterion
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    utils.mkdir('checkpoints')
    # Load ckpt file
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        if opts.distributed:
            model = DDP(model)
        else:
            model = nn.DataParallel(model)
        model.to(device)
        if local_rank == opts.default_rank:
            logger.info("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print(opts.ckpt)
        raise Exception("checkpoint file error! please check")

    # ==========   Test Loop   ==========#
    model.eval()
    val_score, ret_samples, _ = validate(
        opts=opts, model=model, loader=test_loader, device=device, metrics=metrics, local_rank=local_rank,
        criterion=criterion, distributed=opts.distributed)
    if local_rank == opts.default_rank:
        logger.info(metrics.to_str(val_score))


if __name__ == '__main__':
    main()

