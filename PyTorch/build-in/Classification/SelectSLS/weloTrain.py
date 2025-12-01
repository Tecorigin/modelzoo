#!/usr/bin/env python3
# coding: utf-8
"""
通用训练模版（优先从本地导入 Model -> 支持 DDP / 单卡，AMP，resume，日志，checkpoint）
保存为 train_template_localmodel.py
"""
import os
import sys
import time
import json
import argparse
from collections import OrderedDict
from pathlib import Path
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import importlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as tv_models

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch.sdaa import amp
# from torch.cuda import amp


# ----------------------------
# Helper utilities (self-contained)
# ----------------------------
class AverageMeter(object):
    def __init__(self, name='Meter', fmt=':.4f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} (avg {avg' + self.fmt + '})'
        return fmtstr.format(name=self.name, val=self.val, avg=self.avg)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    返回一个 list，每个元素是 tensor（百分比形式）
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # output: (N, C) -> pred: (maxk, N)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()  # (maxk, N)
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # (maxk, N) bool

        res = []
        for k in topk:
            # 把前 k 行展平后求和（返回 0-dim tensor），随后换算为百分比
            correct_k = correct[:k].reshape(-1).float().sum()  # 注意：不传 keepdim
            # 乘以 100.0 / batch_size，保持返回 tensor（和之前代码兼容）
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth'):
    save_path = os.path.join(save_dir, filename)
    torch.save(state, save_path)
    if is_best:
        best_path = os.path.join(save_dir, 'model_best.pth')
        torch.save(state, best_path)

def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True

# ----------------------------
# Argument parser
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Generic PyTorch training template (DDP/AMP) with LocalModel priority')
    parser.add_argument('--name', default='run', type=str, help='experiment name (log/checkpoints dir)')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--arch', default='None', type=str, help='model name')
    parser.add_argument('--deterministic', action='store_true', help='set cudnn deterministic (may be slower)')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10','cifar100','imagenet','custom'], help='which dataset')
    parser.add_argument('--datapath', default='./data', type=str, help='dataset root / imagenet root / custom root')
    parser.add_argument('--imagenet_dir', default='./imagenet', type=str, help='if dataset=imagenet, path to imagenet root')
    parser.add_argument('--custom_eval_dir', default=None, help='if dataset=custom, provide val dir')
    parser.add_argument('--num_workers', default=4, type=int, help='dataloader workers per process')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--model_name', default='resnet18', help='torchvision model name or python path e.g. mypkg.mymodule.Model (used if no local Model)')
    parser.add_argument('--num_classes', default=None, type=int, help='override num classes (auto-detect for common sets)')
    parser.add_argument('--pretrained', action='store_true', help='use torchvision pretrained weights when available')
    parser.add_argument('--optimizer', default='sgd', choices=['sgd','adam','adamw'], help='optimizer')
    parser.add_argument('--lr', '--learning_rate', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--scheduler', default='multistep', choices=['multistep','step','cosine','none'], help='lr scheduler')
    parser.add_argument('--milestones', default='100,150', type=str, help='milestones for multistep (comma sep)')
    parser.add_argument('--step_size', default=30, type=int, help='step size for StepLR or cosine max epochs')
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--scheduler_step_per_batch', action='store_true', help='call scheduler.step() per batch (for some schedulers)')
    parser.add_argument('--resume', default='', type=str, help='path to checkpoint to resume from')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--save_freq', default=10, type=int, help='save checkpoint every N epochs (rank0 only)')
    parser.add_argument('--amp', action='store_true', default = True,help='use automatic mixed precision (AMP)')
    parser.add_argument('--grad_accum_steps', default=1, type=int, help='gradient accumulation steps')
    parser.add_argument('--local_rank', default=None, type=int, help='local rank passed by torchrun (if any). Use -1 or None for non-distributed')
    parser.add_argument('--cutmix_prob', default=0.0, type=float)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--seed_sampler', default=False, action='store_true', help='set sampler epoch seeds to make deterministic distributed shuffling')
    args = parser.parse_args()
    args.milestones = [int(x) for x in args.milestones.split(',')] if args.milestones else []
    return args

# ----------------------------
# build model (优先 LocalModel)
# ----------------------------
def build_model_with_local_priority(args, device=None):
    """
    用参数 args.arch 作为模块名导入 Model()
    如果模块不存在或没有 Model 类，则报错停止。
    """
    try:
        # 动态导入模块，比如 args.arch = "rexnet"
        mod = importlib.import_module(args.arch)
        Model = getattr(mod, "Model")   # 从模块中获取 Model 类
    except Exception as e:
        raise RuntimeError(
            f"无法导入模型模块 '{args.arch}' 或未找到类 Model。"
            f"\n错误信息：{e}"
        )
    
    # 解析数据集类别数
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    else:
        print(f"[ERROR] 不支持的数据集类型：{args.dataset}，无法确定类别数。程序终止。")
        sys.exit(1)


    # 实例化
    try:
        model = Model(num_classes)
    except Exception as e:
        raise RuntimeError(
            f"Model() 实例化失败，请检查模型构造函数。\n错误信息：{e}"
        )

    return model

# ----------------------------
# Data loader factory
# ----------------------------
def build_dataloaders(args, rank, world_size):
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616) if args.dataset == 'cifar10' else (0.2023, 0.1994, 0.2010)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        root = args.datapath
        if args.dataset == 'cifar10':
            train_set = datasets.CIFAR10(root=root, train=True, download=False, transform=train_transform)
            val_set = datasets.CIFAR10(root=root, train=False, download=False, transform=test_transform)
            num_classes = 10
        else:
            train_set = datasets.CIFAR100(root=root, train=True, download=False, transform=train_transform)
            val_set = datasets.CIFAR100(root=root, train=False, download=False, transform=test_transform)
            num_classes = 100

    elif args.dataset == 'imagenet':
        train_dir = os.path.join(args.imagenet_dir, 'train')
        val_dir = os.path.join(args.imagenet_dir, 'val')
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
        ])
        train_set = datasets.ImageFolder(train_dir, train_transform)
        val_set = datasets.ImageFolder(val_dir, test_transform)
        num_classes = args.num_classes or 1000

    elif args.dataset == 'custom':
        train_dir = os.path.join(args.datapath, 'train')
        val_dir = args.custom_eval_dir or os.path.join(args.datapath, 'val')
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        train_set = datasets.ImageFolder(train_dir, train_transform)
        val_set = datasets.ImageFolder(val_dir, test_transform)
        num_classes = len(train_set.classes)
    else:
        raise ValueError("Unknown dataset")

    if dist.is_initialized() and world_size > 1:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=(train_sampler is None),
                              num_workers=args.num_workers,
                              pin_memory=True,
                              sampler=train_sampler,
                              drop_last=False)
    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True)

    return train_loader, val_loader, num_classes, train_sampler

# ----------------------------
# Train & validate
# ----------------------------
def train_one_epoch(args, epoch, model, criterion, optimizer, train_loader, device, scaler, scheduler=None, train_sampler=None):
    batch_time = AverageMeter('Time')
    data_time = AverageMeter('Data')
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')

    model.train()
    end = time.time()
    optimizer.zero_grad()

    iters = len(train_loader)
    for i, (images, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if args.amp:
            with amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets) / args.grad_accum_steps
        else:
            outputs = model(images)
            loss = criterion(outputs, targets) / args.grad_accum_steps

        if args.amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % args.grad_accum_steps == 0:
            if args.amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None and args.scheduler_step_per_batch:
                scheduler.step()

        with torch.no_grad():
            acc1, acc5 = accuracy(outputs, targets, topk=(1,5))
        losses.update(loss.item() * args.grad_accum_steps, images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        # if (i % args.print_freq == 0) and ((dist.get_rank() if dist.is_initialized() else 0) == 0):
        #     lr = optimizer.param_groups[0]['lr']
        #     print(f"Epoch[{epoch}] Iter[{i}/{iters}] lr={lr:.6f} loss={losses.val:.4f} avg={losses.avg:.4f} "
        #           f"top1={top1.val:.2f} avg={top1.avg:.2f} time={batch_time.val:.3f}s")

    if scheduler is not None and not args.scheduler_step_per_batch:
        scheduler.step()

    return OrderedDict([('loss', losses.avg), ('acc1', top1.avg), ('acc5', top5.avg)])

def validate(args, model, val_loader, criterion, device):
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')

    model.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(val_loader)):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, topk=(1,5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
    return OrderedDict([('loss', losses.avg), ('acc1', top1.avg), ('acc5', top5.avg)])

# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()

    # handle local_rank from env if not provided
    local_rank_env = os.environ.get('LOCAL_RANK', None)
    if args.local_rank is None and local_rank_env is not None:
        args.local_rank = int(local_rank_env)

    distributed = (args.local_rank is not None and args.local_rank != -1)
    if distributed:
        dist.init_process_group(backend='nccl', init_method='env://')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_seed(args.seed + (rank if distributed else 0), deterministic=args.deterministic)

    save_dir = os.path.join('models', args.name)
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
    if distributed:
        dist.barrier()

    train_loader, val_loader, auto_num_classes, train_sampler = build_dataloaders(args, rank, world_size)
    if args.num_classes is None:
        args.num_classes = auto_num_classes

    # 使用本地 Model 优先（LocalModel 已在文件顶部尝试导入）
    model = build_model_with_local_priority(args, device)
    model.to(device)

    if distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    criterion = nn.CrossEntropyLoss().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError('Unknown optimizer')

    scheduler = None
    if args.scheduler == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'none':
        scheduler = None

    scaler = amp.GradScaler() if args.amp else None

    start_epoch = args.start_epoch
    best_acc = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            ckpt = torch.load(args.resume, map_location='cpu')
            model_state = ckpt.get('state_dict', ckpt)
            if isinstance(model, DDP):
                model.module.load_state_dict(model_state)
            else:
                model.load_state_dict(model_state)
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt.get('epoch', start_epoch)
            best_acc = ckpt.get('best_acc', best_acc)
            print(f"=> resumed from {args.resume}, start_epoch={start_epoch}")
        else:
            print(f"=> resume path {args.resume} not found")

    log_columns = ['epoch', 'lr', 'loss', 'acc1', 'acc5', 'val_loss', 'val_acc1', 'val_acc5']
    log_df = pd.DataFrame(columns=log_columns)

    total_epochs = args.epochs
    for epoch in range(start_epoch, total_epochs):
        if train_sampler is not None:
            if args.seed_sampler:
                train_sampler.set_epoch(epoch + args.seed)
            else:
                train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"==== Epoch {epoch}/{total_epochs - 1} ====")

        train_log = train_one_epoch(args, epoch, model, criterion, optimizer, train_loader, device, scaler, scheduler, train_sampler)
        val_log = validate(args, model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]['lr']

        if rank == 0:
            print("Epoch[{}]: train_loss {:.4f} acc1 {:.2f} acc5 {:.2f} | val_loss {:.4f} acc1 {:.2f} acc5 {:.2f} lr {:.6f}".format(
                epoch, train_log['loss'], train_log['acc1'], train_log['acc5'],
                val_log['loss'], val_log['acc1'], val_log['acc5'], current_lr
            ))
            row = {
                'epoch': epoch,
                'lr': current_lr,
                'loss': train_log['loss'],
                'acc1': train_log['acc1'],
                'acc5': train_log['acc5'],
                'val_loss': val_log['loss'],
                'val_acc1': val_log['acc1'],
                'val_acc5': val_log['acc5'],
            }
            # log_df = log_df.append(row, ignore_index=True)
            new_row_df = pd.DataFrame([row])
            log_df = pd.concat([log_df, new_row_df], ignore_index=True)
            log_df.to_csv(os.path.join(save_dir, 'log.csv'), index=False)

            is_best = val_log['acc1'] > best_acc
            if is_best:
                best_acc = val_log['acc1']
            if (epoch % args.save_freq == 0) or is_best or (epoch == total_epochs - 1):
                state = {
                    'epoch': epoch,
                    'state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'args': vars(args)
                }
                save_checkpoint(state, is_best, save_dir, filename=f'checkpoint_epoch_{epoch}.pth')

    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        print("Training finished. Best val acc1: {:.2f}".format(best_acc))

def set_global_seed(seed: int = 42, set_threads: bool = True):
    """
    统一设置随机性，保证 CUDA / AMP / CPU 在不同机器上的结果尽量可复现
    （适用于多卡、自研算子、AMP 训练场景）

    Args:
        seed (int): 随机种子
        set_threads (bool): 是否限制线程数（建议 True，保证可重复性）
    """

    # --------------------------
    # 1. 环境变量
    # --------------------------
    os.environ["PYTHONHASHSEED"] = str(seed)

    # OpenMP / MKL 线程固定（避免非确定性）
    if set_threads:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

    # --------------------------
    # 2. Python / Numpy RNG
    # --------------------------
    random.seed(seed)
    np.random.seed(seed)

    # --------------------------
    # 3. PyTorch RNG
    # --------------------------
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多卡适用（包含自研卡）

    # --------------------------
    # 4. 强制确定性
    # --------------------------
    # 重要：AMP 不影响 determinism，但某些算子在 AMP 下会切换实现
    # 这里确保使用确定性版本
    torch.use_deterministic_algorithms(True)

    # CUDNN 相关（兼容 CUDA / SDAA 等）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --------------------------
    # 5. 固定 PyTorch 内部线程数
    # --------------------------
    if set_threads:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    print(f"[Seed Utils] 已设置全局随机种子 = {seed}")

if __name__ == '__main__':
    set_global_seed(2025)    # 任意数字都行
    main()