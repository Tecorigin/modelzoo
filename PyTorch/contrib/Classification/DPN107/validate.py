from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time
import torch_sdaa,torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as data
import torch_sdaa.amp  as amp           

import model_factory
from dataset import Dataset
import torch.multiprocessing as mp
import torch.distributed as dist
import warnings

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', '-m', metavar='MODEL', default='dpn107',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=224, type=int,
                    metavar='N', help='Input image dimension')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--multi-gpu', dest='multi_gpu', action='store_true',
                    help='use multiple-gpus')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool for DPN models')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:65501', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

def ma():
    #os.environ['MASTER_PORT'] = '8888'  # 端口号随意指定

    args = parser.parse_args()

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    ngpus_per_node = torch.sdaa.device_count()
    if args.multi_gpu:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main(args.gpu, ngpus_per_node, args)

def main(gpu, ngpus_per_node, args):
    # 修改后，使用SDAA作为训练设备：
    if args.multi_gpu:
        device = torch.device('sdaa:{}'.format(args.gpu))
    else:
        device = torch.device('sdaa')
    # 修改前的CUDA接口:
    # torch.cuda.is_available()

    # 改为SDAA接口：
    print(torch.sdaa.is_available())

    # 修改前的CUDA接口：
    # torch.cuda.device_count()

    # 修改后CUDA接口：
    torch.sdaa.device_count()
    #args = parser.parse_args()

    test_time_pool = False
    if 'dpn' in args.model and args.img_size > 224 and not args.no_test_pool:
        test_time_pool = True

    if not args.checkpoint and not args.pretrained:
        args.pretrained = True  # might as well do something...

    # create model
    num_classes = 1000
    model = model_factory.create_model(
        args.model,
        num_classes=num_classes,
        pretrained=False,
        test_time_pool=test_time_pool)
    
    

    print('Model %s created, param count: %d' %
          (args.model, sum([m.numel() for m in model.parameters()])))

    # optionally resume from a checkpoint
    if args.checkpoint and os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("=> loaded checkpoint '{}'".format(args.checkpoint))
    elif args.checkpoint:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
        exit(1)

    args.gpu = gpu
    if args.multi_gpu:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.gpu:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        torch.sdaa.set_device(args.gpu)    
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        #print(model.device)                        
        model.to('sdaa:{}'.format(args.gpu))
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    else:
        model.to('sdaa')

    #model = torch.nn.DataParallel(model).sdaa()
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion=criterion.to('sdaa')

    cudnn.benchmark = True

    transforms = model_factory.get_transforms_eval(
        args.model,
        args.img_size)

    dataset = Dataset(
        args.data,
        transforms)

    if args.multi_gpu:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = None
    
    if args.multi_gpu:
        loader = data.DataLoader(
            dataset,
            batch_size=args.batch_size, num_workers=args.workers,shuffle=False
            ,pin_memory=True, sampler=train_sampler)
    else:
        loader = data.DataLoader(
            dataset,
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    # model.eval()
    # end = time.time()
    # with torch.no_grad():
    #     for i, (input, target) in enumerate(loader):
    #         target = target.cuda()
    #         input = input.cuda()

    #         # compute output
    #         output = model(input)
    #         loss = criterion(output, target)

    #         # measure accuracy and record loss
    #         prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    #         losses.update(loss.item(), input.size(0))
    #         top1.update(prec1.item(), input.size(0))
    #         top5.update(prec5.item(), input.size(0))

    #         # measure elapsed time
    #         batch_time.update(time.time() - end)
    #         end = time.time()

    #         if i % args.print_freq == 0:
    #             print('Test: [{0}/{1}]\t'
    #                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
    #                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
    #                 i, len(loader), batch_time=batch_time, loss=losses,
    #                 top1=top1, top5=top5))

    # print(' * Prec@1 {top1.avg:.3f} ({top1a:.3f}) Prec@5 {top5.avg:.3f} ({top5a:.3f})'.format(
    #     top1=top1, top1a=100-top1.avg, top5=top5, top5a=100.-top5.avg))



    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 切换到训练模式
    model.train()
    #model = torch.DataParallel(model) # 开启数据并行模式
    # 训练过程
    #model = torch.nn.DataParallel(model)  # 使用 DataParallel
    
    scaler = amp.GradScaler()              # 添加GradScaler
    end = time.time()


    import matplotlib.pyplot as plt

    # 用于保存每一步的损失值
    losses = []

    # 创建保存图像的文件夹（如果不存在）
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # 设置绘图环境
    plt.ion()  # 打开交互模式
    fig, ax = plt.subplots()  # 创建一个图形对象和一个子图

    # 设置图的标签
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')

    for epoch in range(2):
        for i, (input, target) in enumerate(loader):

            input = input.to('sdaa',non_blocking=True)
            target = target.to('sdaa',non_blocking=True)
            input = input.to(memory_format=torch.channels_last) # 数据格式转换为NHWC（channels_last）
            # print(f"Model is on device: {next(model.parameters()).device}")
            # print(f"Input is on device: {input.device}")
            print("设备设置")
            optimizer.zero_grad()  # 清空梯度
            with amp.autocast():
                print("完成清空梯度")
                output = model(input)  # 前向传播
                print("完成前向传播")
                loss = criterion(output, target)  # 计算损失
            print("完成损失计算")
            scaler.scale(loss).backward()
            print("完成损失反向传播")
            scaler.step(optimizer)
            print("完成优化器step")
            scaler.update()
            # scaler.scale(loss).backward()    # loss缩放并反向转播
            # scaler.step(optimizer)    # 更新参数（自动unscaling）
            # scaler.update()    # 更新loss_scaling系数


            # 保存当前损失值
            losses.append(loss.item())
            
            # 更新loss图
            ax.plot(losses, label='Loss', color='blue')
            ax.legend()
            plt.draw()  # 更新图像
            plt.pause(0.1)  # 暂停一下以更新图表

            # 保存图像到文件
            plt.savefig(os.path.join(output_dir, f'loss_epoch{epoch}_step{i}.png'))
            # 打印训练进度
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            print(f"Epoch [{epoch}/{2}], Step [{i}/{len(loader)}], Loss: {loss.item()}, Prec@1: {prec1.item()}, Prec@5: {prec5.item()}")




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    ma()
