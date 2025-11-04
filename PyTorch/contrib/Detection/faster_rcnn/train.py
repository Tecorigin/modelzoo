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
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.frcnn import FasterRCNN
from nets.frcnn_training import (FasterRCNNTrainer, get_lr_scheduler,
                                 set_optimizer_lr, weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import FRCNNDataset, frcnn_dataset_collate
from utils.utils import (get_classes, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch
import torch_sdaa
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
                        default="faster_rcnn", type=str, help="name of the model")
    parser.add_argument("--epoch", required=False, default=10, type=int, help="number of total epochs to run")
    parser.add_argument("--batch_size", "--bs", required=False, default=8, type=int,
                        help="mini-batch size (default: 64) per device")
    parser.add_argument("--nproc_per_node", required=False, default=1, type=int,
                        help="The number of processes to launch on each node, "
                             "for GPU training, this is recommended to be set "
                             "to the number of GPUs in your system so that "
                             "each process can be bound to a single GPU.")
    parser.add_argument('--device', required=False, default='sdaa', type=str,
                        help='which device to use. cuda, sdaa optional, sdaa default')
    parser.add_argument("--use_amp", required=False, default=False, type=str2bool, help='Distributed training or not')
    parser.add_argument("--use_ddp", required=False, default=False, type=str2bool, help='DDP training or not')
    parser.add_argument("--local-rank", required=False, default=-1, type=int)
    parser.add_argument('--classes_path', required=False, default='model_data/voc_classes.txt', type=str)
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
    classes_path = args.classes_path

    # model_path = ''
    model_path = 'model_data/voc_weights_vgg.pth'
    input_shape = [600, 600]
    backbone = 'vgg'
    pretrained = True
    anchors_size = [8, 16, 32]

    Init_Epoch = 0
    Init_lr = 1e-7
    Min_lr = Init_lr * 0.01

    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0
    lr_decay_type = 'cos'
    save_period = 1  # 多少个epoch保存一次权值
    save_dir = 'logs'
    num_workers = 0
    # train_annotation_path = os.path.join(project_path,'2007_train.txt')
    # val_annotation_path = os.path.join(project_path,'2007_val.txt')
    train_annotation_path = '2007_train.txt'
    val_annotation_path = '2007_val.txt'

    class_names, num_classes = get_classes(classes_path)

    seed_everything(seed)

    model = FasterRCNN(num_classes, anchor_scales=anchors_size, backbone=backbone, pretrained=pretrained)

    if not pretrained:
        weights_init(model)

    if model_path != '':
        print('Load weights {}.'.format(model_path))

        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

    model = model.to(device)
    model_train = model.train()

    if use_ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, find_unused_parameters=True)
    else:
        model = model

    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    if use_amp:
        print('use_amp')
        scaler = torch_sdaa.amp.GradScaler()  # 定义GradScaler
    else:
        print('not use_amp')
        scaler = None

    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()[:1000]
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()[:300]
    num_train = len(train_lines)
    num_val = len(val_lines)
    if local_rank == 0:
        show_config(
            classes_path=classes_path, model_path=model_path, input_shape=input_shape, \
            epoch=Epoch, batch_size=batch_size, \
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum, lr_decay_type=lr_decay_type, \
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )

    if True:
        for param in model.module.extractor.parameters():
            param.requires_grad = False
        for param in model.module.head.parameters():
            param.requires_grad = False
        model.module.freeze_bn()
        nbs = 16
        lr_limit_max = 1e-4 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                             weight_decay=weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        train_dataset = FRCNNDataset(train_lines, input_shape, train=True)
        val_dataset = FRCNNDataset(val_lines, input_shape, train=False)

        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=frcnn_dataset_collate,
                         worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=frcnn_dataset_collate,
                             worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

        train_util = FasterRCNNTrainer(model_train, optimizer)

        for epoch in range(Init_Epoch, Epoch):

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(device, model, train_util, loss_history, optimizer, epoch, epoch_step, epoch_step_val,
                          gen, gen_val, Epoch, use_sdaa, use_amp, scaler, save_period, save_dir, batch_size, local_rank)

        if local_rank == 0:
            loss_history.writer.close()
