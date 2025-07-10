# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# Adapted to tecorigin hardware。
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

from torch_sdaa.utils import cuda_migrate  # 使用torch_sdaa自动迁移方法
from torch.sdaa import amp              # 导入AMP

from tcap_dllogger import Logger, StdOutBackend,    JSONStreamBackend, Verbosity

json_logger = Logger(
    [
        StdOutBackend(Verbosity.DEFAULT),
        JSONStreamBackend(Verbosity.VERBOSE,    'dlloger_example.json'),
    ]
)
json_logger.metadata("train.loss", {"unit": "", "GOAL":    "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.ips",{"unit": "imgs/s",    "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    # metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()

    from time import time
    _time = time()
        
    for step, (samples, targets) in enumerate(data_loader):
        if step > args.steps:
            break
        
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
        
        with amp.autocast():
            outputs = model(samples)
            if not args.cosub:
                loss = criterion(samples, outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets) 
                loss = loss + 0.25 * criterion(outputs[1], targets) 
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        # loss_scaler(loss, optimizer, clip_grad=max_norm,
        #             parameters=model.parameters(), create_graph=is_second_order)
        loss_scaler.scale(loss).backward(create_graph=is_second_order)

        # 梯度裁剪（如果启用了梯度裁剪）
        if max_norm is not None:
            loss_scaler.unscale_(optimizer)  # 反缩放梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # 进行梯度裁剪

        # 更新优化器
        loss_scaler.step(optimizer)
        loss_scaler.update()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        # 计算ips
        ips = samples.size(0) / (time() - _time)
        _time = time()

        json_logger.log(
            step=(epoch, step),
            data={
                "rank": os.environ.get("LOCAL_RANK", "0"),
                "train.loss": loss_value,
                "train.ips": ips,
            },
            verbosity=Verbosity.DEFAULT,
        )


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.sdaa.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
