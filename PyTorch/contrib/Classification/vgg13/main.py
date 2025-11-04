import argparse
import os
import time

import torch
import torch_sdaa
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
from torch_sdaa import amp

from models import *
from data_loader import data_loader
from helper import AverageMeter, save_checkpoint, accuracy, adjust_learning_rate
# os.environ["SDAA_VISIBLE_DEVICES"] = "0,1"
# os.environ["SDAA_DISTRIBUTED_MODE"] = "1"
# os.environ["TCCL_NCCL_ALGO"] = "RING"


model_names = [
    'alexnet',
    'densenet121', 'densenet169', 'densenet201', 'densenet161',
    'vgg11', 'vgg13', 'vgg16', 'vgg19', 
    'resnet34', 'resnet152'
]

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='alexnet', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: alexnet)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='numer of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='Weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-m', '--pin-memory', dest='pin_memory', action='store_true',
                    help='use pin memory')
parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--print-freq', '-f', default=100, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoitn, (default: None)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

best_prec1 = 0.0

global train_losses, train_top1_accuracies
train_losses = []
train_top1_accuracies = []

def main():
    global args, best_prec1
    args = parser.parse_args()
    

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    try:
        torch.sdaa.set_device(local_rank)
        
        torch.distributed.init_process_group(
            backend="tccl",
            init_method="env://",
            world_size=world_size,
            rank=local_rank
        )
        
        device = torch.device("sdaa")
        
    except Exception as e:
        print(f"初始化分布式训练时发生错误: {str(e)}")
        raise

    # Start time of the training
    start_time = time.time()
    max_duration = 2 * 60 * 60  # Maximum duration in seconds

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))

    if args.arch == 'alexnet':
        model = alexnet(pretrained=args.pretrained)
    elif args.arch == 'densenet121':
        model = densenet121(pretrained=args.pretrained)
    elif args.arch == 'densenet169':
        model = densenet169(pretrained=args.pretrained)
    elif args.arch == 'densenet201':
        model = densenet201(pretrained=args.pretrained)
    elif args.arch == 'densenet161':
        model = densenet161(pretrained=args.pretrained)
    elif args.arch == 'vgg11':
        model = vgg11(pretrained=args.pretrained)
    elif args.arch == 'vgg13':
        model = vgg13(pretrained=args.pretrained)
    elif args.arch == 'vgg16':
        model = vgg16(pretrained=args.pretrained)
    elif args.arch == 'vgg19':
        model = vgg19(pretrained=args.pretrained)
    elif args.arch == 'resnet34':
        model = resnet34(pretrained=args.pretrained)
    elif args.arch == 'resnet152':
        model = resnet152(pretrained=args.pretrained)
    else:
        raise NotImplementedError

    # use sdaa
    # model.sdaa()
    model = model.to(device)
    
    if world_size > 1:
        for param in model.parameters():
            if param.device != device:
                param.data = param.data.to(device)
                
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        torch.sdaa.current_stream().synchronize()
        
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=None,
            output_device=None,
            broadcast_buffers=False,
            find_unused_parameters=False
        )

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # optionlly resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True

    # Data loading
    train_loader, val_loader = data_loader(args.data, args.batch_size, args.workers, args.pin_memory)

    if args.evaluate:
        validate(val_loader, model, criterion, args.start_epoch, args.print_freq, start_time, max_duration,device)
        return

    # 添加调试信息
    print(f"Process rank {local_rank} using device: {device}")
    print(f"World size: {world_size}")
    print(f"SDAA_VISIBLE_DEVICES: {os.environ.get('SDAA_VISIBLE_DEVICES')}")

    for epoch in range(args.start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        if time.time() - start_time > max_duration:
            print("Maximum allowed running time reached. Stopping training.")
            break
        
        adjust_learning_rate(optimizer, epoch, args.lr)

        # train for one epoch

        avg_train_loss, avg_train_top1 = train(train_loader, model, criterion, optimizer, epoch, args.print_freq, start_time, max_duration,device)
        print('loss, acc: ', avg_train_loss, avg_train_top1)
        # evaluate on validation set
        prec1, prec5 = validate(val_loader, model, criterion, epoch, args.print_freq, start_time, max_duration, device)

        # remember the best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        train_losses.append(avg_train_loss)
        train_top1_accuracies.append(avg_train_top1)

        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'arch': args.arch,
        #     'state_dict': model.state_dict(),
        #     'best_prec1': best_prec1,
        #     'optimizer': optimizer.state_dict()
        # }, is_best, args.arch + '.pth')

        loss_save_path = f'./scripts/train_sdaa_3rd_{args.arch}.png'
        plot_losses(train_losses, epoch, loss_save_path)
        
        # acc_save_path = './scripts/accuracy.png'
        # plot_accuracies(train_top1_accuracies, epoch, acc_save_path)



def train(train_loader, model, criterion, optimizer, epoch, print_freq, start_time, max_duration,device):
    scaler = torch.amp.GradScaler('sdaa')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        if time.time() - start_time > max_duration:
            print("Maximum allowed running time reached. Ending this epoch early.")
            break
        
        # measure data loading time
        data_time.update(time.time() - end)

        # 确保数据在正确的设备上
        with torch.sdaa.stream(torch.sdaa.current_stream()):
            input = input.to(device)
            target = target.to(device)
            
            with amp.autocast():
                output = model(input)
                loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
    return losses.avg, top1.avg.item()

def validate(val_loader, model, criterion, epoch, print_freq, start_time, max_duration,device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if time.time() - start_time > max_duration:
            print("Maximum allowed running time reached. Ending this epoch early.")
            break
        # 修改数据传输方式
        with torch.sdaa.stream(torch.sdaa.current_stream()):
            input = input.to(device)
            target = target.to(device)
            
            with torch.no_grad():
                with amp.autocast():
                    output = model(input)
                    loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                # Add current batch's loss and accuracy to the lists
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                # Plot and save Loss and Accuracy for this epoch after each batch


    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return top1.avg, top5.avg

def plot_losses(train_losses, epoch, save_path):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    
    # Save the figure to the specified path instead of showing it
    plt.savefig(save_path)
    plt.close()

def plot_accuracies(train_accuracies, epoch, save_path):
    epochs = range(1, len(train_accuracies) + 1)  # Epochs start from 1
    
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    
    # Save the figure to the specified path instead of showing it
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    main()