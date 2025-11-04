# Adapted to tecorigin hardware。

from __future__ import print_function
from __future__ import division
import argparse
import os
import time

import torch
import torch_sdaa
import torch.utils.data
import torch.optim
import torchvision.transforms as transforms
# import torch.backends.cudnn as cudnn  # [修改] 移除，对应接口在 SDAA 中无效
# cudnn.benchmark = True               # [修改] 移除

import torch.distributed as dist  # [修改] 用于分布式

import net
from dataset import ImageList
import lfw_eval
import layer

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CosFace')

# DATA
parser.add_argument('--root_path', type=str, default='',
                    help='path to root path of images')
parser.add_argument('--database', type=str, default='WebFace',
                    help='Which Database for train. (WebFace, VggFace2)')
parser.add_argument('--train_list', type=str, default=None,
                    help='path to training list')
parser.add_argument('--batch_size', type=int, default=512,
                    help='input batch size for training (default: 512)')
parser.add_argument('--is_gray', type=bool, default=False,
                    help='Transform input image to gray or not  (default: False)')
# Network
parser.add_argument('--network', type=str, default='sphere20',
                    help='Which network for train. (sphere20, sphere64, LResNet50E_IR)')
# Classifier
parser.add_argument('--num_class', type=int, default=None,
                    help='number of people(class)')
parser.add_argument('--classifier_type', type=str, default='MCP',
                    help='Which classifier for train. (MCP, AL, L)')
# LR policy
parser.add_argument('--epochs', type=int, default=30,
                    help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--step_size', type=list, default=None,
                    help='lr decay step')  # [15000, 22000, 26000][80000,120000,140000][100000, 140000, 160000]
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    metavar='W', help='weight decay (default: 0.0005)')
# Common settings
parser.add_argument('--log_interval', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_path', type=str, default='/data/ckpt/CosFace_pretrain',
                    help='path to save checkpoint')
parser.add_argument('--workers', type=int, default=4,
                    help='how many workers to load data')

# [修改] 新增分布式相关的显式参数(可选)，也支持从os.environ获取
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--world_size', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()

# [修改] 根据环境变量确认是否为分布式模式
ddp_training = False
if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    ddp_training = True
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.gpu = int(os.environ['LOCAL_RANK'])

if ddp_training:
    # [修改] 分布式场景，多卡
    device = torch.device(f"sdaa:{args.gpu}")
    torch.sdaa.set_device(device)  # [修改] 设置 SDAA 设备
    dist.init_process_group(backend='tccl', init_method='env://',
                            rank=args.rank, world_size=args.world_size)  # [修改] 使用 TCCL
else:
    device = torch.device("sdaa")

if args.database == 'WebFace':
    args.train_list = "/data/datasets/CosFace/CASIA-WebFace-112X96.txt" \
        if args.train_list is None else args.train_list
    args.num_class = 10572 if args.num_class is None else args.num_class
    args.step_size = [16000, 24000] if args.step_size is None else args.step_size
elif args.database == 'VggFace2':
    args.train_list = '/home/wangyf/dataset/VGG-Face2/VGG-Face2-112X96.txt' \
        if args.train_list is None else args.train_list
    args.num_class = 8069 if args.num_class is None else args.num_class
    args.step_size = [80000, 120000, 140000] if args.step_size is None else args.step_size
else:
    raise ValueError("NOT SUPPORT DATABASE! ")


def main():
    # --------------------------------------model----------------------------------------
    if args.network == 'sphere20':
        model_single = net.sphere(type=20, is_gray=args.is_gray)
        model_eval = net.sphere(type=20, is_gray=args.is_gray)
    elif args.network == 'sphere64':
        model_single = net.sphere(type=64, is_gray=args.is_gray)
        model_eval = net.sphere(type=64, is_gray=args.is_gray)
    elif args.network == 'LResNet50E_IR':
        model_single = net.LResNet50E_IR(is_gray=args.is_gray)
        model_eval = net.LResNet50E_IR(is_gray=args.is_gray)
    else:
        raise ValueError("NOT SUPPORT NETWORK! ")

    # [修改] 在分布式下需要先将模型放到device，再构建DDP
    model_single.to(device)
    if ddp_training:
        # [修改] 使用 DistributedDataParallel
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model_single, device_ids=[args.gpu], output_device=args.gpu)
    else:
        # [修改] 与原逻辑一致，DataParallel 仅在非DDP时可考虑
        model = torch.nn.DataParallel(model_single).to(device)

    model_eval = model_eval.to(device)
    # [修改] 不打印模型信息
    # print(model)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # [修改] 模型保存路径保持一致
    if not ddp_training or (ddp_training and args.rank == 0):
        # 只有 rank0 负责保存
        model_single.save(os.path.join(args.save_path, 'CosFace_0_checkpoint.pth'))

    # 512 is dimension of feature
    # [修改] 先创建 classifier_single，再在分布式时包成 DDP
    classifier_single = {
        'MCP': layer.MarginCosineProduct(512, args.num_class).to(device),
        'AL': layer.AngleLinear(512, args.num_class).to(device),
        'L': torch.nn.Linear(512, args.num_class, bias=False).to(device)
    }[args.classifier_type]

    if ddp_training:
        from torch.nn.parallel import DistributedDataParallel as DDP
        classifier = DDP(classifier_single, device_ids=[args.gpu], output_device=args.gpu)
    else:
        classifier = classifier_single

    # ------------------------------------load image---------------------------------------
    if args.is_gray:
        train_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])  # gray
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])

    # [修改] 如果分布式训练，需要使用 DistributedSampler
    if ddp_training:
        from torch.utils.data.distributed import DistributedSampler
        train_dataset = ImageList(root=args.root_path, fileList=args.train_list,
                                  transform=train_transform)
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,  # [修改] 分布式时要由 sampler 控制 shuffle
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            ImageList(root=args.root_path, fileList=args.train_list,
                      transform=train_transform),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True
        )

    # [修改] 仅在 rank 0 或非分布式时打印数据集信息，避免多卡重复
    if not ddp_training or (ddp_training and args.rank == 0):
        print('length of train Database: ' + str(len(train_loader.dataset)))
        print('Number of Identities: ' + str(args.num_class))

    # --------------------------------loss function and optimizer-----------------------------
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD([{'params': model.parameters()},
                                 {'params': classifier.parameters()}],
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # [修改] 使用 AMP 的 GradScaler
    scaler = torch.sdaa.amp.GradScaler()

    # ----------------------------------------train----------------------------------------
    for epoch in range(1, args.epochs + 1):
        # [修改] 分布式场景下，需要设置 sampler 的 epoch
        if ddp_training:
            train_loader.sampler.set_epoch(epoch)

        train(train_loader, model, classifier, criterion, optimizer, scaler, epoch)
        if not ddp_training or (ddp_training and args.rank == 0):
            model_single.save(os.path.join(args.save_path, f'CosFace_{epoch}_checkpoint.pth'))
            lfw_eval.eval(model_eval, os.path.join(args.save_path, f'CosFace_{epoch}_checkpoint.pth'), args.is_gray)

    # [修改] 仅在 rank 0 或非分布式时打印数据集信息，避免多卡重复
    if not ddp_training or (ddp_training and args.rank == 0):
        print('Finished Training')

    # [修改] 如果是分布式，则结束后销毁进程组
    if ddp_training:
        dist.destroy_process_group()


def train(train_loader, model, classifier, criterion, optimizer, scaler, epoch):
    model.train()
    print_with_time(f'Epoch {epoch} start training', rank=args.rank, only_master=True)
    time_curr = time.time()
    loss_display = 0.0

    for batch_idx, (data, target) in enumerate(train_loader, 1):
        iteration = (epoch - 1) * len(train_loader) + batch_idx
        adjust_learning_rate(optimizer, iteration, args.step_size)

        # [修改] 数据拷贝到 SDAA 并转 NHWC (channels_last)
        data = data.to(device).to(memory_format=torch.channels_last)
        target = target.to(device)
        # compute output
        with torch.sdaa.amp.autocast():  # [修改] 在 autocast 环境中前向
            output = model(data)
            if isinstance(classifier, torch.nn.Linear):
                output = classifier(output)
            else:
                output = classifier(output, target)
            loss = criterion(output, target)

        loss_display += loss.item()
        # compute gradient and do SGD step
        optimizer.zero_grad()
        # [修改] 使用 scaler 做混合精度反向
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if batch_idx % args.log_interval == 0:
            time_used = time.time() - time_curr
            loss_display /= args.log_interval
            # [修改] 为避免 AttributeError，需要判断 classifier 是否有 'module'
            if args.classifier_type == 'MCP':
                if hasattr(classifier, 'module'):
                    INFO = f' Margin: {classifier.module.m:.4f}, Scale: {classifier.module.s:.2f}'
                else:
                    INFO = f' Margin: {classifier.m:.4f}, Scale: {classifier.s:.2f}'
            elif args.classifier_type == 'AL':
                if hasattr(classifier, 'module'):
                    INFO = f' lambda: {classifier.module.lamb:.4f}'
                else:
                    INFO = f' lambda: {classifier.lamb:.4f}'
            else:
                INFO = ''
            print_with_time(
                'Train Epoch: {} [{}/{} ({:.0f}%)]{}, Loss: {:.6f}, Elapsed time: {:.4f}s({} iters)'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    iteration, loss_display, time_used, args.log_interval) + INFO,
                args.rank,
                False
            )
            time_curr = time.time()
            loss_display = 0.0


def print_with_time(msg, rank=0, only_master=True):
    # [修改] 在多卡情况下区分打印方式
    if only_master:
        if rank == 0:
            print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()) + msg)
    else:
        # 如果想让所有rank都打印，可以在日志前加[Rank x]
        print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()) + f"[Rank {rank}] " + msg + "\n")


def adjust_learning_rate(optimizer, iteration, step_size):
    """Sets the learning rate to the initial LR decayed by 10 each step size"""
    if iteration in step_size:
        lr = args.lr * (0.1 ** (step_size.index(iteration) + 1))
        print_with_time('Adjust learning rate to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        pass


if __name__ == '__main__':
    # [修改] 仅在 rank 0 或非分布式时打印数据集信息，避免多卡重复
    if not ddp_training or (ddp_training and args.rank == 0):
        print(args)
    main()