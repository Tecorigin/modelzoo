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

import matplotlib.pyplot as plt
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity


from models import LinkNet
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

    parser.add_argument("--work_dir", type=str, default=None, 
                        help="specify the working directory, example:experiments/linknet, \
                             if not specified, wor_dir will be like:exp2, exp3, exp4")

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='vaihingen',
                        choices=['vaihingen', ], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=6,
                        help="num classes (default: None)")


    parser.add_argument("--model_name", type=str, default='linknet',
                        choices=['linknet', ], help='model name')

    # Train Options
    parser.add_argument("--distributed", type=str2bool, default=False)
    parser.add_argument("--use_amp", type=str2bool, default=False)
    parser.add_argument("--default_rank",  type=int, default=0)
    parser.add_argument("--total_epochs", type=int, default=100,
                        help="epoch number (default: 30k)")
    parser.add_argument("--optimizer", type=str, default='sgd', choices=['sgd', 'adam', 'adamw'],
                        help="optimizer for training")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='cosine', choices=['poly', 'step', 'cosine'],
                        help="learning rate scheduler policy")
    parser.add_argument("--crop_val", type=str2bool, default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help='batch size for validation (default: 1)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", type=str2bool, default=False)
    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument('--device', required=True, default='sdaa', type=str,
                        help='which device to use. cuda, sdaa optional, sdaa default')
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=5,
                        help="print interval of loss (default: 5)")
    parser.add_argument("--val_epoch", type=int, default=5,
                        help="epoch interval for eval (default: 5)")

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
    
    # Setup logger and working directory
    if local_rank == opts.default_rank:
        if opts.work_dir:
            work_dir = opts.work_dir
        else:
            work_dir = get_new_experiment_folder()

        # get current time for log name
        current_time = datetime.now()
        # format time information
        time_str = current_time.strftime("%Y%m%d_%H%M%S")

        json_logger = Logger(
        [
            # StdOutBackend(Verbosity.DEFAULT),
            JSONStreamBackend(Verbosity.VERBOSE, os.path.join(work_dir, f'dlloger_{time_str}.json')),
        ]
        )

        json_logger.metadata("train.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
        json_logger.metadata("val.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "VALID"})
        json_logger.metadata("train.ips",{"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})
        json_logger.metadata("val.ips",{"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "VALID"})
        json_logger.metadata("train.compute_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
        json_logger.metadata("train.fp_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
        json_logger.metadata("train.bp_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
        json_logger.metadata("train.grad_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})


        logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join(work_dir, f"train_{time_str}.log")),
                        logging.StreamHandler()
                    ])
        logger = logging.getLogger("TecoSeg")
        opts_dict = vars(opts)
        formatted_opts = json.dumps(opts_dict, indent=4)
        logger.info("============arguments=============")
        logger.info(formatted_opts)
        logger.info("==================================")
        logger.info("Main Device: [%s]" % device)

    # Init processgroup, choose tccl as backend
    if opts.distributed:
        torch.distributed.init_process_group(backend="tccl", init_method="env://")
        if local_rank == opts.default_rank:
            logger.info("process backend succeessfully initialized: [tccl]")

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_dst, val_dst, _, CLASS_NAMES = get_dataset(opts)
    
    # Setup Dataloader 
    if opts.distributed:
        # Distrubuted dataloader
        train_sampler = DistributedSampler(train_dst)
        val_sampler = DistributedSampler(val_dst)

        train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size, 
                                    sampler=train_sampler, num_workers=0, drop_last=True)

        val_loader = data.DataLoader(val_dst, batch_size=opts.val_batch_size, 
                                    sampler=val_sampler, num_workers=0)
    else:
        train_loader = data.DataLoader(
            train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=0,
            drop_last=True)  # drop_last=True to ignore single-image batches.
        val_loader = data.DataLoader(
            val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=0)
    
    if local_rank == opts.default_rank:
        logger.info("Dataset: [%s] loaded , Train set: [%d], Val set: [%d]"
                     % (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model
    if opts.model_name == "linknet":
        model = LinkNet(opts.num_classes).to(device)
    else:
        raise Exception("Sorry, the model you choose is not supported now")

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes, CLASS_NAMES)

    # Setup scaler
    scaler = torch_sdaa.amp.GradScaler() # sdaa need this

    # Set up optimizer
    if opts.optimizer == "sgd": 
        optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    elif opts.optimizer == "adamw": 
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    elif opts.optimizer == "adam": 
        optimizer = torch.optim.Adam(params=model.parameters(), lr=opts.lr)
    
    # Setup learning rate policy
    total_itrs = opts.total_epochs * len(train_loader)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(opts.total_epochs // 5) * steps_per_epoch, gamma=0.1)
    elif opts.lr_policy == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_itrs, eta_min=opts.lr * 0.01)

    # Set up criterion
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    
    def save_ckpt(path, ddp=False):
        """ save current model
        """
        torch.save({
                "cur_epochs": cur_epochs,
                "model_state": model.module.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_score": best_score,
            }, path)

        if local_rank == opts.default_rank:
            logger.info("Model saved as %s" % path)


    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 1
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_epochs = checkpoint["cur_epochs"]
            best_score = checkpoint['best_score']
            if local_rank == opts.default_rank:
                logger.info("Training state restored from %s" % opts.ckpt)
        if local_rank == opts.default_rank:
            logger.info("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        if local_rank == opts.default_rank:
            logger.info("[--- Start Retraining ---]")

    
    if opts.distributed:
        # model = torch_sdaa.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, find_unused_parameters=True)
        model.to(device)
    else:
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#

    interval_loss = 0
    if local_rank == opts.default_rank:
        train_losses = []
        val_losses = []
        epoch_numbers = []  # 用于存储每次验证时的epoch数

    while True:
        model.train()
        # model = model.to(memory_format=torch.channels_last)
        epoch_loss = 0.0
        num_iterations = len(train_loader)
        
        cur_itrs = 0
        for idx, (images, labels) in enumerate(train_loader):
            cur_itrs += 1

            start_time = time.time()

            if opts.use_amp:
                # images = images.to(device, dtype=torch.float32).to(memory_format=torch.channels_last)
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                data_time = time.time() - start_time
                compute_start_time = time.time()

                with torch.sdaa.amp.autocast():   # 开启AMP环境
                    fp_start_time = time.time()

                    outputs = model(images)
                    
                    loss = 0
                    if type(outputs) is tuple:  # output = (main_loss, aux_loss1, axu_loss2***)
                        length = len(outputs)
                        for index, out in enumerate(outputs):
                            loss_record = criterion(out, labels)
                            if index == 0:
                                loss_record *= 0.6
                            else:
                                loss_record *= 0.4 / (length - 1)
                            loss += loss_record
                        outputs = outputs[0]
                    else:
                        loss = criterion(outputs, labels)

                    fp_time = time.time() - fp_start_time
                optimizer.zero_grad()

                bp_start_time = time.time()
                scaler.scale(loss).backward()    # loss缩放并反向传播
                bp_time = time.time() - bp_start_time

                grad_start_time = time.time()
                scaler.step(optimizer)    # 参数更新
                scaler.update()    # 基于动态Loss Scale更新loss_scaling系数
                grad_time = time.time() - grad_start_time
            else:
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                data_time = time.time() - start_time
                compute_start_time = time.time()

                fp_start_time = time.time()
                outputs = model(images) # [bs, class_num, h, w]
                
                loss = 0
                if type(outputs) is tuple or type(outputs) is list:  # output = (main_loss, aux_loss1, axu_loss2***)
                    length = len(outputs)
                    for index, out in enumerate(outputs):
                        loss_record = criterion(out, labels)
                        if index == 0:
                            loss_record *= 0.6
                        else:
                            loss_record *= 0.4 / (length - 1)
                        loss += loss_record
                    outputs = outputs[0]
                else:
                    loss = criterion(outputs, labels)

                fp_time = time.time() - fp_start_time

                optimizer.zero_grad()

                bp_start_time = time.time()
                scaler.scale(loss).backward()    # loss缩放并反向传播
                bp_time = time.time() - bp_start_time

                grad_start_time = time.time()
                scaler.step(optimizer)    # 参数更新
                scaler.update()    # 基于动态Loss Scale更新loss_scaling系数
                grad_time = time.time() - grad_start_time

            compute_time = time.time() - compute_start_time
            

            np_loss = loss.detach().cpu().numpy()
            epoch_loss += np_loss
            interval_loss += np_loss

            # json logger
            if local_rank == opts.default_rank:
                batch_size = images.size(0)
                ips = batch_size / (time.time() - start_time)
                json_logger.log(
                    step = (cur_epochs, cur_epochs * num_iterations + cur_itrs),
                    data = {
                            "rank":local_rank,
                            "train.loss":np_loss, 
                            "output.shape":outputs.shape,
                            "train.ips":ips,
                            "train.lr":optimizer.param_groups[0]['lr'],
                            "train.data_time":data_time,
                            "train.compute_time":compute_time,
                            "train.fp_time":fp_time,
                            "train.bp_time":bp_time,
                            "train.grad_time":grad_time,
                            },
                    verbosity=Verbosity.DEFAULT,
                )

                # 更新训练损失并保存图像
                train_losses.append(np_loss)
                plot_train_loss(train_losses, work_dir)
            
            # 每个 iteration 的日志输出
            if (idx + 1) % opts.print_interval == 0:
                interval_loss = interval_loss / opts.print_interval 
                if local_rank == opts.default_rank:
                    current_lr = optimizer.param_groups[0]['lr']
                    if local_rank == opts.default_rank:
                        logger.info(f"Epoch [{cur_epochs}/{opts.total_epochs}], "
                                    f"Itr [{idx + 1}/{num_iterations}], "
                                    f"Loss: {interval_loss:.6f}, "
                                    f"LR: {current_lr:.6e}")
                interval_loss = 0.0
            
            # 每个 epoch 的日志输出
            if (cur_itrs) == num_iterations and local_rank == opts.default_rank:
                epoch_loss = epoch_loss / num_iterations
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Epoch [{cur_epochs}/{opts.total_epochs}], "
                            f"Average Train Loss: {epoch_loss:.6f}")

            if (cur_epochs) % opts.val_epoch == 0 and (cur_itrs) == num_iterations:
                if local_rank == opts.default_rank:
                    save_ckpt(os.path.join(work_dir, f'latest_{opts.model_name}_{opts.dataset}.pth'), opts.distributed)
                    logger.info("Starting validation...")
                val_start_time = time.time()

                model.eval()
                val_score, ret_samples, val_loss = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, local_rank=local_rank, criterion=criterion,
                    distributed=opts.distributed)
                val_time = time.time() - val_start_time
                val_ips = len(val_loader.dataset) / val_time

                if local_rank == opts.default_rank:
                    logger.info("Validation results:\n" + metrics.to_str(val_score))

                # Save best model
                if val_score['Mean IoU'] > best_score:  
                    best_score = val_score['Mean IoU']
                    if local_rank == opts.default_rank:
                        save_ckpt(os.path.join(work_dir, f'best_{opts.model_name}_{opts.dataset}.pth'), opts.distributed)
                
                if local_rank == opts.default_rank:
                    json_logger.log(
                        step = (cur_epochs, cur_epochs * num_iterations + cur_itrs),
                        data = {
                            "val.loss":val_loss,
                            "val.ips":val_time,
                            "val.metric":val_score
                        },
                        verbosity=Verbosity.DEFAULT,
                    )
                    # 更新验证损失和epoch数，并保存图像
                    val_losses.append(val_loss)
                    epoch_numbers.append(cur_epochs)
                    plot_val_loss(val_losses, epoch_numbers, work_dir)

                model.train()
            scheduler.step()

        cur_epochs += 1
        
        if opts.distributed:
            dist.barrier()
        
        if cur_epochs > opts.total_epochs:
            break
    

if __name__ == '__main__':
    main()
