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
import torch_sdaa
import os
from tqdm import tqdm
from utils import ext_transforms as et
from datasets import Vaihingen_VOC
import torch.distributed as dist

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'vaihingen':
        CLASS_NAMES = ['Impervious surfaces', 'Building', 'Low vegetation',
                       'Tree', 'Car', 'Clutter']
        train_transform = et.ExtCompose([
            # et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            # et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            # et.ExtRandomRotation(degrees=30),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = Vaihingen_VOC(root=opts.data_root, image_set='train', transform=train_transform)
        val_dst = Vaihingen_VOC(root=opts.data_root, image_set='val', transform=val_transform)
        test_dst = Vaihingen_VOC(root=opts.data_root, image_set='val', transform=val_transform)
    else:
        raise Exception("Sorry, the dataset type you choose is not supported now")

    return train_dst, val_dst, test_dst, CLASS_NAMES

def validate(opts, model, loader, device, metrics, local_rank, criterion, distributed=False):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    num_batches = len(loader)  # 计算batch的数量

    with torch.no_grad():
        if distributed and local_rank != opts.default_rank:
            data_iter = enumerate(loader)
        else:
            data_iter = tqdm(enumerate(loader), total=len(loader))
        
        val_loss = 0.0
            
        for i, (images, labels) in data_iter:

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            val_loss += criterion(outputs, labels).detach().cpu().numpy()
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
        
        if distributed:
            # All-reduce to gather confusion matrices from all processes
            if dist.is_initialized():
                dist.barrier()  # 同步所有进程，确保所有进程都完成了验证
                metrics.all_reduce_confusion_matrix(device=device)

        score = metrics.get_results()

        avg_val_loss = val_loss / num_batches

    return score, ret_samples, avg_val_loss
