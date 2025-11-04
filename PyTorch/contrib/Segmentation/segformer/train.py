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

import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.segformer import SegFormer
from nets.segformer_training import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import SegmentationDataset, seg_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch
import torch_sdaa
import random
import argparse
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=False,
                        default="segformer", type=str, help="name of the model")
    parser.add_argument("--epoch", required=False, default=100, type=int, help="number of total epochs to run")
    parser.add_argument("--batch_size", "--bs", required=False, default=8, type=int,
                        help="mini-batch size (default: 64) per device")
    parser.add_argument("--nproc_per_node", required=False, default=1, type=int,
                        help="The number of processes to launch on each node, "
                             "for GPU training, this is recommended to be set "
                             "to the number of GPUs in your system so that "
                             "each process can be bound to a single GPU.")
    parser.add_argument('--device', required=False, default='sdaa', type=str,
                        help='which device to use. cuda, sdaa optional, sdaa default')
    parser.add_argument("--use_amp", required=False, default=True, type=str2bool)
    parser.add_argument("--use_ddp", required=False, default=False, type=str2bool, help='DDP training or not')
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument('--dataset_path', required=False, default='VOCdevkit', type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name
    Epoch = args.epoch
    batch_size = args.batch_size
    nproc_per_node = args.nproc_per_node
    use_ddp = args.use_ddp
    if use_ddp:
        local_rank = args.local_rank
        # DDP backend初始化
        device = torch.device(f"sdaa:{local_rank}")
        torch.sdaa.set_device(device)
        # 初始化ProcessGroup，通信后端选择tccl
        torch.distributed.init_process_group(backend="tccl", init_method="env://")
    else:
        local_rank = 0
        device = args.device
    use_amp = args.use_amp
    use_sdaa = True
    seed = 11

    num_classes = 21  # 自己需要的分类个数+1
    phi = "b0"  # b0、b1、b2、b3、b4、b5

    pretrained = False
    model_path = ''
    input_shape = [512, 512]

    Init_Epoch = 0
    Init_lr = 1e-4  # 1e-4
    Min_lr = Init_lr * 0.01
    optimizer_type = "adamw"
    momentum = 0.9
    weight_decay = 1e-2
    lr_decay_type = 'cos'
    save_period = 5  # 多少个epoch保存一次权值
    save_dir = 'logs'
    VOCdevkit_path = args.dataset_path
    dice_loss = False

    focal_loss = False
    cls_weights = np.ones([num_classes], np.float32)

    num_workers = 0

    seed_everything(seed)

    model = SegFormer(device=device, num_classes=num_classes, phi=phi, pretrained=pretrained)

    if not pretrained:
        weights_init(model)

    if use_ddp:
        model = DDP(model)
    else:
        model = model

    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    if use_amp:
        scaler = torch_sdaa.amp.GradScaler()  # 定义GradScaler
    else:
        scaler = None


    model_train = model.train()
    model_train = model_train.to(device)

    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        all_train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        all_val_lines = f.readlines()

    train_lines = random.sample(all_train_lines, 1500)
    val_lines = random.sample(all_val_lines, 200)
    # train_lines = all_train_lines
    # val_lines = all_val_lines
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes=num_classes, phi=phi, model_path=model_path, input_shape=input_shape,
            Init_Epoch=Init_Epoch,
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type,
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )

    if True:
        nbs = 16
        lr_limit_max = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
        lr_limit_min = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr, betas=(momentum, 0.999), weight_decay=weight_decay),
            'adamw': optim.AdamW(model.parameters(), Init_lr, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr, momentum=momentum, nesterov=True,
                             weight_decay=weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, Epoch)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        train_dataset = SegmentationDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset = SegmentationDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

        train_sampler = None
        val_sampler = None
        shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=seg_dataset_collate, sampler=train_sampler,
                         worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=seg_dataset_collate, sampler=val_sampler,
                             worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

        for epoch in range(Init_Epoch, Epoch):
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(device, model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val,
                          gen, gen_val, Epoch, use_sdaa, dice_loss, focal_loss, cls_weights, num_classes, use_amp, scaler, save_period, save_dir,
                          local_rank, batch_size)
        if local_rank == 0:
            loss_history.writer.close()
