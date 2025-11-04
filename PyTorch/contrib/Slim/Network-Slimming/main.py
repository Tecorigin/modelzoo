# Adapted to tecorigin hardware。

from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch_sdaa
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import torch.distributed as dist  # [修改] 用于DDP分布式训练
from vgg import vgg

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables SDAA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='/data/ckpt/Network-Slimming', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--depth', default=19, type=int,
                    help='depth of the neural network')

args = parser.parse_args()

# [修改] 判断是否具备SDAA设备可用
args.sdaa_available = torch.sdaa.is_available()
args.use_sdaa = (not args.no_cuda) and args.sdaa_available

# [修改] 从环境变量获取分布式相关参数（配合 torchrun）
if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    args.rank = int(os.environ['RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.gpu = int(os.environ['LOCAL_RANK'])
else:
    # 单机或未设置分布式环境时的默认值
    args.rank = 0
    args.world_size = 1
    args.gpu = 0

torch.manual_seed(args.seed)

# [修改] 设定训练使用的 device
if args.use_sdaa:
    # 当使用分布式并且有多卡时，指定 local_rank 作为索引
    device = torch.device(f"sdaa:{args.gpu}")
    # [修改] 设置默认设备为 SDAA
    torch.sdaa.set_device(device)
else:
    device = torch.device("cpu")

# [修改] 若为多进程分布式训练，初始化进程组
if args.world_size > 1:
    dist.init_process_group(backend='tccl', init_method='env://',
                            rank=args.rank, world_size=args.world_size)

if not os.path.exists(args.save):
    os.makedirs(args.save, exist_ok=True)

# 准备数据集
kwargs = {'num_workers': 1, 'pin_memory': True} if args.use_sdaa else {}
if args.dataset == 'cifar10':
    train_dataset = datasets.CIFAR10('/data/datasets/Network-Slimming', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                (0.2023, 0.1994, 0.2010))
                       ]))
    test_dataset = datasets.CIFAR10('/data/datasets/Network-Slimming', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                (0.2023, 0.1994, 0.2010))
                       ]))
else:
    train_dataset = datasets.CIFAR100('/data/datasets/Network-Slimming', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                (0.2023, 0.1994, 0.2010))
                       ]))
    test_dataset = datasets.CIFAR100('/data/datasets/Network-Slimming', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                (0.2023, 0.1994, 0.2010))
                       ]))

# [修改] 当进行分布式训练时，使用 DistributedSampler
if args.world_size > 1:
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True
    )
else:
    train_sampler = None

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    **kwargs
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.test_batch_size,
    shuffle=True,
    **kwargs
)

# 构建模型
if args.refine:
    checkpoint = torch.load(args.refine, map_location='cpu')
    model = vgg(dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
    model.load_state_dict(checkpoint['state_dict'])
else:
    model = vgg(dataset=args.dataset, depth=args.depth)

# [修改] 模型放到SDAA设备
model.to(device)

# [修改] 若多卡分布式，使用DDP包裹模型
if args.world_size > 1:
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu], output_device=args.gpu
    )

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if args.resume:
    if os.path.isfile(args.resume):
        if args.rank == 0:
            print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if args.rank == 0:
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                  .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        if args.rank == 0:
            print("=> no checkpoint found at '{}'".format(args.resume))

# [修改] AMP 相关
# 在太初 SDAA 上进行 float16 优化，需要使用 torch.sdaa.amp
scaler = torch.sdaa.amp.GradScaler() if args.use_sdaa else None


# additional subgradient descent on the sparsity-induced penalty term
def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s * torch.sign(m.weight.data))  # L1

def train(epoch):
    model.train()
    if args.world_size > 1 and train_sampler is not None:
        # [修改] 分布式训练时，为保证随机性，每个 epoch 都要设置一次 sampler
        train_sampler.set_epoch(epoch)

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device, memory_format=torch.channels_last)
        target = target.to(device)

        optimizer.zero_grad()

        # [修改] 在 AMP autocast 环境下前向与 loss 计算
        if args.use_sdaa:
            with torch.sdaa.amp.autocast():
                output = model(data)
                loss = F.cross_entropy(output, target)
        else:
            # CPU 情况下，不做 autocast
            output = model(data)
            loss = F.cross_entropy(output, target)

        # [修改] 使用scaler进行反向传播和更新
        if scaler:
            scaler.scale(loss).backward()

            # 先解除梯度缩放（Unscale），再进行手动 L1 正则化
            scaler.unscale_(optimizer)
            if args.sr:
                updateBN()

            # 继续正常的优化步骤
            scaler.step(optimizer)
            scaler.update()
        else:
            # 纯 FP32 训练路径
            loss.backward()
            if args.sr:
                updateBN()
            optimizer.step()

        if batch_idx % args.log_interval == 0 and args.rank == 0:
            # [修改] loss使用item()
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test():
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        if args.use_sdaa:
            # 推理时也可使用 autocast，提高推理效率
            for data, target in test_loader:
                data = data.to(device, memory_format=torch.channels_last)
                target = target.to(device)
                with torch.sdaa.amp.autocast():
                    output = model(data)
                    test_loss += F.cross_entropy(output, target, reduction='sum').item()
                    pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
        else:
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    if args.rank == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

def save_checkpoint(state, is_best, filepath):
    if args.rank == 0:
        torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

best_prec1 = 0.
for epoch in range(args.start_epoch, args.epochs):
    if epoch in [int(args.epochs*0.5), int(args.epochs*0.75)]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    train(epoch)
    prec1 = test()
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best, filepath=args.save)

if args.rank == 0:
    print("Best accuracy: {:.4f}".format(best_prec1))