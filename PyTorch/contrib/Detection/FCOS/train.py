# Adapted to tecorigin hardware。

import os
import json
import argparse
import time
import math
from argparse import ArgumentParser, ArgumentTypeError

import torch
import torch_sdaa
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import random
import numpy as np
from utils import train_one_epoch, evaluate, get_datasets
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt

from model.fcos import FCOSDetector
from dataset.COCO_dataset import COCODataset
from dataset.augment import Transforms

local_rank = int(os.environ.get("LOCAL_RANK", -1))

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


def main(args):
    if args.distributed is False:
        device = torch.device(args.device)
    else:
        device = torch.device(f"sdaa:{local_rank}")
        torch.sdaa.set_device(device)
        torch.distributed.init_process_group(backend="tccl",init_method="env://")


    dataset_path = args.dataset_path
    val_dataset_path = args.dataset_path.replace('train', 'val')
    annotation_path = args.annotation_path
    val_annotation_path = args.annotation_path.replace('train', 'val')
    
    img_size = 224
    
    transform = Transforms()
    train_dataset = COCODataset(dataset_path, annotation_path, transform=transform)
    val_dataset = COCODataset(val_dataset_path, val_annotation_path)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    if local_rank != -1:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=(train_sampler is None),
                                               pin_memory=True,
                                               num_workers=8,
                                               sampler=train_sampler,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=(train_sampler is None),
                                             pin_memory=True,
                                             num_workers=8,
                                             sampler=val_sampler,
                                             collate_fn=train_dataset.collate_fn)

    model = FCOSDetector(mode="training")
    model.to(args.device)

    if args.distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)
 
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scaler = torch_sdaa.amp.GradScaler()

    best_acc = 0.
    global_step = 0

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    
    for epoch in range(args.epochs):
        if local_rank != -1:
            train_sampler.set_epoch(epoch)
        start_time = time.time()
        train_throughput = len(train_loader.dataset)
        train_loss, train_acc, train_data_to_device_time, train_compute_time, total_forward_time, total_backward_time, total_optimizer_step_time = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                scaler=scaler,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                use_acm = args.autocast,
                                                rank = opt.local_rank,
                                                local_rank = local_rank,
                                                img_size=img_size,
                                                lr=args.lr,
                                                train_throughput=train_throughput,
                                                max_step=args.step,
                                                save_path=args.path)
        scheduler.step()
        if local_rank == 0 or local_rank == -1:
            with open(os.path.join(args.path, 'log_epoch.jsonl'), 'a') as f:
                json.dump({'Epoch': epoch, 'Loss': float(train_loss), 'Acc': float(train_acc)}, f)
                f.write('\n')
            if 'scripts' in os.getcwd():
                with open(os.path.join(os.getcwd(), '../log_epoch.jsonl'), 'a') as f:
                    json.dump({'Epoch': epoch, 'Loss': float(train_loss), 'Acc': float(train_acc)}, f)
                    f.write('\n') 
            else:
                with open(os.path.join(os.getcwd(), 'log_epoch.jsonl'), 'a') as f:
                    json.dump({'Epoch': epoch, 'Loss': float(train_loss), 'Acc': float(train_acc)}, f)
                    f.write('\n')
        end_time = time.time()
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnodes", default=1, type=int)
    parser.add_argument("--local-rank", default= -1, type=int)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--distributed', type=str2bool, default=True)
    parser.add_argument('--autocast', type=str2bool, default=True)
    parser.add_argument("--step", default=-1, type=int)
    parser.add_argument('--dataset_path', type=str, default="")
    parser.add_argument('--annotation_path', type=str, default='')

    parser.add_argument('--freeze_layers', type=bool, default=False)
    parser.add_argument('--device', default='sdaa')
    parser.add_argument('--path', type=str, default='/data/ckpt/FCOS/experiments/')

    opt = parser.parse_args()
    
    opt.str_lr = str(opt.lr).replace('.', '_')
    if not opt.distributed:
        opt.path = '/data/ckpt/FCOS/single_experiments/'
    opt.path = os.path.join(opt.path, f'batchsize_{opt.batch_size}_lr_{opt.str_lr}')
    os.makedirs(opt.path, exist_ok=True)
    if 'scripts' in os.getcwd():
        with open(os.path.join(os.getcwd(), '../log_epoch.jsonl'), 'w') as f:
            pass
        with open(os.path.join(os.getcwd(), '../log.jsonl'), 'w') as f:
            pass
    else:
        with open(os.path.join(os.getcwd(), 'log_epoch.jsonl'), 'w') as f:
            pass
        with open(os.path.join(os.getcwd(), 'log.jsonl'), 'w') as f:
            pass
    
    local_rank = opt.local_rank

    main(opt)