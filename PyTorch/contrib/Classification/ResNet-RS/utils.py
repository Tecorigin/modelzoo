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
from time import time

import loguru
import torch

# 初始化logger
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
from torchvision import transforms
from torchvision.datasets import ImageFolder

json_logger = Logger([
    StdOutBackend(Verbosity.DEFAULT),
    JSONStreamBackend(Verbosity.VERBOSE, "log.json"),
])
json_logger.metadata("train.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.ips", {"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_datasets(dataset_path):
    train_dataset = ImageFolder(os.path.join(dataset_path, "train"), train_transforms)
    val_dataset = ImageFolder(os.path.join(dataset_path, "val"), val_transforms)

    return train_dataset, val_dataset


def collate_fn(batch):
    images, labels = tuple(zip(*batch))

    images = torch.stack(images, dim=0)
    labels = torch.as_tensor(labels)
    return images, labels


def train_one_epoch(model, loss_function, optimizer, scaler, data_loader, device, epoch, args):
    model.train()
    optimizer.zero_grad()

    global_step = 0

    _time = time()
    loguru.logger.info(f"开始训练迭代")
    for step, data in enumerate(data_loader):
        if 0 < args.max_step <= global_step:
            break
        images, labels = data

        if "sdaa" in args.device:
            images, labels = images.to(device, dtype=args.dtype).to(memory_format=torch.channels_last), labels.to(device)
        else:
            images, labels = images.to(device, dtype=args.dtype), labels.to(device)

        if args.amp and "sdaa" in args.device:
            with torch.sdaa.amp.autocast():
                pred = model(images)
                loss = loss_function(pred, labels)
        else:
            pred = model(images)
            loss = loss_function(pred, labels)

        # 计算ips
        batch_size = images.shape[0]
        ips = batch_size / (time() - _time)
        _time = time()

        json_logger.log(
            step=(epoch, step),
            data={
                "rank": args.local_rank,
                "train.loss": loss.item(),
                "train.ips": ips,
            },
            verbosity=Verbosity.DEFAULT,
        )

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()  # loss缩放并反向传播
            scaler.step(optimizer)  # 参数更新
            scaler.update()  # 基于动态Loss Scale更新loss_scaling系数
        else:
            loss.backward()
            optimizer.step()

        if not torch.isfinite(loss):
            loguru.logger.error("non-finite loss, ending training ")
            sys.exit(1)
        global_step += 1


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, args):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device, dtype=args.dtype))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device, dtype=args.dtype)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
