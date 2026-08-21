#!/usr/bin/env python3
# coding: utf-8

import os
import random
import sys
import time
import json
import argparse
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import importlib
from inplace_abn import InPlaceABN

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"     # 强烈推荐在 shell/最顶端设置
os.environ["PYTHONHASHSEED"] = "12345"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def ensure_cublas_workspace(config=":4096:8"):
    """
    尝试为 cuBLAS 设置可复现 workspace。强烈建议在主脚本入口处（import torch 之前）
    通过 export 设置该 env。此函数会在运行时设置，但如果 torch 已经被 import，
    则可能为时已晚——函数会打印提醒。
    """
    already = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if already:
        print(f"[seed_utils] CUBLAS_WORKSPACE_CONFIG 已存在：{already}")
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = config
        print(f"[seed_utils] 已设置 CUBLAS_WORKSPACE_CONFIG={config} （注意：请在 import torch 前设置以保证生效）")

def set_global_seed(seed: int = 42, set_threads: bool = True):
    """
    统一随机性设置。注意：若希望完全发挥效果，请在主脚本入口（import torch 之前）
    先调用 ensure_cublas_workspace(...) 或在 shell 中 export CUBLAS_WORKSPACE_CONFIG。
    """
    ensure_cublas_workspace()  # 会设置 env 并提醒
    os.environ["PYTHONHASHSEED"] = str(seed)

    if set_threads:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

    random.seed(seed)
    np.random.seed(seed)

    # 现在导入 torch（晚导入以便前面 env 生效）
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 强制确定性（如果存在不确定性算子，PyTorch 会报错并提示）
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print("[seed_utils] 设置 deterministic 模式时出错：", e)
        print("[seed_utils] 请确认 CUBLAS_WORKSPACE_CONFIG 已在 import torch 之前设置。")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if set_threads:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    print(f"[seed_utils] 全局 seed 已设置为 {seed}")

set_global_seed(2025) 

"""
通用训练模版（优先从本地导入 Model -> 支持 DDP / 单卡，AMP，resume，日志，checkpoint）
保存为 train_template_localmodel.py
"""
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
    parser.add_argument('--steps', default=0, type=int, help='max steps to run (if >0, training will stop when global_step reaches this).')
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
        # train_transform = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean, std),
        # ])
        # test_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean, std),
        # ])

        train_transform = transforms.Compose([    # 2025/12/3 从visformer模型开始
        transforms.Resize(256),                 # 先放大到 256
        transforms.RandomCrop(224),            # 再随机裁剪为 224（更符合 ImageNet 风格增强）
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
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
def train_one_epoch(args, epoch, model, criterion, optimizer, train_loader, device, scaler, scheduler=None, train_sampler=None, global_step_start=0, max_global_steps=None):
    """
    现在支持：若 max_global_steps 非 None，则当 global_step 达到该值时提前退出
    返回: epoch_summary_dict, step_logs_list, global_step_end
    step_logs_list: list of dicts with per-step info (for logging to CSV if需要)
    """
    batch_time = AverageMeter('Time')
    data_time = AverageMeter('Data')
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')

    model.train()
    end = time.time()
    optimizer.zero_grad()

    iters = len(train_loader)
    step_logs = []
    global_step = global_step_start

    for i, (images, targets) in enumerate(train_loader):
        # check global steps limit
        if (max_global_steps is not None) and (global_step >= max_global_steps):
            break

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

        # 每当累积步满足 grad_accum_steps 就 step
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

        # increment global step AFTER processing this batch
        global_step += 1

        # per-step print (controlled by print_freq)
        if ((global_step % args.print_freq == 0) or (i == iters - 1)) and ((dist.get_rank() if dist.is_initialized() else 0) == 0):
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch[{epoch}]:step[{i+1}/{iters}] step_train_loss {losses.val:.4f} acc1 {top1.val:.2f} acc5 {top5.val:.2f}")

        # collect per-step log
        step_logs.append({
            'epoch': epoch,
            'batch_idx': i,
            'global_step': global_step,
            'lr': optimizer.param_groups[0]['lr'],
            'loss': losses.val,
            'loss_avg': losses.avg,
            'acc1': top1.val,
            'acc1_avg': top1.avg,
            'acc5': top5.val,
            'acc5_avg': top5.avg,
            'time': batch_time.val
        })

        # if reached max_global_steps inside epoch, break (handled at loop start next iter)
        if (max_global_steps is not None) and (global_step >= max_global_steps):
            if (dist.get_rank() if dist.is_initialized() else 0) == 0:
                print(f"[Info] 达到 max_global_steps={max_global_steps}，将在 epoch 内提前停止。")
            break

    # --- flush remaining grads if needed (handle gradient accumulation leftovers) ---
    processed_batches = global_step - global_step_start  # 实际处理的 batch 数
    if args.grad_accum_steps > 1 and (processed_batches % args.grad_accum_steps) != 0:
        # only step if there are gradients
        grads_present = any((p.grad is not None and p.requires_grad) for p in model.parameters())
        if grads_present:
            if args.amp:
                try:
                    scaler.step(optimizer)
                    scaler.update()
                except Exception as e:
                    # 防御性：若 scaler.step 因某些原因失败，尝试普通 step（只在极端情况下）
                    print("[Warning] scaler.step 失败，尝试普通 optimizer.step():", e)
                    optimizer.step()
            else:
                optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None and args.scheduler_step_per_batch:
                scheduler.step()
            if (dist.get_rank() if dist.is_initialized() else 0) == 0:
                print(f"[Info] flushed remaining gradients after early stop (processed_batches={processed_batches}, grad_accum={args.grad_accum_steps}).")

    if scheduler is not None and not args.scheduler_step_per_batch:
        scheduler.step()

    return OrderedDict([('loss', losses.avg), ('acc1', top1.avg), ('acc5', top5.avg)]), step_logs, global_step

def validate(args, model, val_loader, criterion, device, max_batches=None):
    """
    Validate on the val_loader.
    If max_batches is not None, only process up to that many batches (useful for quick checks).
    Returns an OrderedDict with loss/acc1/acc5 (averaged over processed samples).
    """
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')

    model.eval()
    processed_batches = 0
    processed_samples = 0
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(val_loader)):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, topk=(1,5))
            batch_n = images.size(0)
            losses.update(loss.item(), batch_n)
            top1.update(acc1.item(), batch_n)
            top5.update(acc5.item(), batch_n)

            processed_batches += 1
            processed_samples += batch_n

            if (max_batches is not None) and (processed_batches >= max_batches):
                break

    # 如果没处理任何样本，避免除0（不太可能，但防御性）
    if processed_samples == 0:
        return OrderedDict([('loss', 0.0), ('acc1', 0.0), ('acc5', 0.0)])
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

    # ==========================================================================
    # [终极方案] 移除 InplaceABN，替换为原生 PyTorch 层
    # 解决所有 Segfault 和驱动不兼容问题
    # ==========================================================================
    if InPlaceABN is not None:
        def replace_iabn_with_standard_layers(module):
            for name, child in module.named_children():
                if isinstance(child, InPlaceABN):
                    # 1. 获取原层参数
                    num_features = child.num_features
                    eps = child.eps
                    momentum = child.momentum
                    affine = child.affine
                    track_running_stats = child.track_running_stats
                    
                    # 2. 获取激活函数配置 (TResNet 通常是用 leaky_relu)
                    act_name = getattr(child, 'activation', 'leaky_relu')
                    act_param = getattr(child, 'activation_param', 0.01)
                    
                    # 3. 创建原生 BatchNorm2d
                    bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, 
                                      affine=affine, track_running_stats=track_running_stats)
                    
                    # 迁移权重 (state_dict key 通常兼容)
                    bn.load_state_dict(child.state_dict(), strict=False)
                    
                    # 4. 创建激活函数 (InplaceABN 是融合层，所以拆分为 BN + Act)
                    layers = [bn]
                    if act_name == 'leaky_relu':
                        layers.append(nn.LeakyReLU(negative_slope=act_param, inplace=True))
                    elif act_name == 'relu':
                        layers.append(nn.ReLU(inplace=True))
                    elif act_name == 'identity' or act_name == 'none':
                        pass # 无激活
                    else:
                        # 默认兜底
                        layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
                    
                    # 5. 替换为 Sequential (BN + Act)
                    # 注意：将层转回当前设备
                    replacement = nn.Sequential(*layers).to(device)
                    setattr(module, name, replacement)
                    
                    if rank == 0:
                        print(f"  -> Replaced {name} with BatchNorm2d + {act_name}")
                else:
                    # 递归处理
                    replace_iabn_with_standard_layers(child)

        print("[Setup] 正在移除 InplaceABN 并替换为标准 PyTorch 层...")
        replace_iabn_with_standard_layers(model)
        if rank == 0:
            print("[Setup] 替换完成。模型现在使用原生算子，硬件兼容性 100%。")
    # ==========================================================================
    
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
    # step-level log
    step_log_columns = ['epoch', 'batch_idx', 'global_step', 'lr', 'loss', 'loss_avg', 'acc1', 'acc1_avg', 'acc5', 'acc5_avg', 'time']
    step_log_df = pd.DataFrame(columns=step_log_columns)

    total_epochs = args.epochs
    # global_step计数器（训练过程中跨epoch持续）
    global_step = 0

    epoch = start_epoch
    # loop until either epoch criteria or step criteria met
    while True:
        if train_sampler is not None:
            if args.seed_sampler:
                train_sampler.set_epoch(epoch + args.seed)
            else:
                train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"==== Epoch {epoch}/{total_epochs - 1} ====")

        # 如果传入了 args.steps (>0)，则把剩余允许的 step 数传给 train_one_epoch，
        # 否则 max_global_steps=None（按整 epoch 执行完）
        if args.steps and args.steps > 0:
            max_global_steps = args.steps
        else:
            max_global_steps = None

        train_log, step_logs, global_step = train_one_epoch(
            args, epoch, model, criterion, optimizer, train_loader, device, scaler,
            scheduler, train_sampler, global_step_start=global_step, max_global_steps=max_global_steps
        )

                # 如果启用了按 steps 的模式且已经达到上限，标记需要在做一次验证后退出
        if max_global_steps is not None and global_step >= max_global_steps:
            if rank == 0:
                print(f"[Main] 达到 max_global_steps={max_global_steps}（global_step={global_step}），将在完成验证后退出训练。")
            # 我们不 return 立刻退出；后面的 validate / 保存逻辑会执行一次，然后 main 返回/结束
            end_due_to_steps = True
        else:
            end_due_to_steps = False

        # 验证并记录 epoch 级别日志（如果在 step 模式下很可能在中间某个 epoch 提前结束，但我们仍做一次 validate）
        val_log = validate(args, model, val_loader, criterion, device, args.batch_size)
        current_lr = optimizer.param_groups[0]['lr']

        if rank == 0:
            # epoch summary print, 格式与示例对齐
            print(f"Epoch[{epoch}]: epoch_train_loss {train_log['loss']:.4f} acc1 {train_log['acc1']:.2f} acc5 {train_log['acc5']:.2f} | "
                  f"val_loss {val_log['loss']:.4f} acc1 {val_log['acc1']:.2f} acc5 {val_log['acc5']:.2f} lr {current_lr:.6f}")
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
            new_row_df = pd.DataFrame([row])
            log_df = pd.concat([log_df, new_row_df], ignore_index=True)
            log_df.to_csv(os.path.join(save_dir, 'log.csv'), index=False)

            is_best = val_log['acc1'] > best_acc
            if is_best:
                best_acc = val_log['acc1']
            if (epoch % args.save_freq == 0) or is_best or ( (max_global_steps is None) and (epoch == total_epochs - 1) ) :
                state = {
                    'epoch': epoch,
                    'state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'args': vars(args)
                }
                save_checkpoint(state, is_best, save_dir, filename=f'checkpoint_epoch_{epoch}.pth')

        # 如果是因为 steps 模式达到上限，则在完成 validation / 保存后退出训练
        if end_due_to_steps:
            if rank == 0:
                print(f"[Main] 已在 steps 模式下完成最后一次验证并保存，训练结束（global_step={global_step}）。")
            break

        # increment epoch
        epoch += 1

        # stopping conditions:
        # 1) if steps mode enabled and reached steps -> stop
        if args.steps and args.steps > 0:
            if global_step >= args.steps:
                if rank == 0:
                    print(f"[Main] 已达到指定 steps={args.steps}（global_step={global_step}），训练结束。")
                break

        # 2) if steps not used, stop when epoch >= epochs
        else:
            if epoch >= total_epochs:
                if rank == 0:
                    print(f"[Main] 已达到指定 epochs={total_epochs}（epoch={epoch}），训练结束。")
                break

    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        print("Training finished. Best val acc1: {:.2f}".format(best_acc))

if __name__ == '__main__':
    main()