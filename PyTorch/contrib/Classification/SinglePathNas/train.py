import argparse
import logging
import os
import sys
import time

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets

import utils
from model import SinglePath_OneShot

from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
from torch.sdaa import amp

scaler = torch.sdaa.amp.GradScaler()    # 定义GradScaler

# 初始化 JSON Logger
json_logger = Logger(
    [
        StdOutBackend(Verbosity.DEFAULT),
        # JSONStreamBackend(Verbosity.VERBOSE, "tmp.json"),
    ]
)

# 设置元数据
json_logger.metadata("train.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.ips", {"unit": "imgs/s", "format": "%.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("val_loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "VAL"})
json_logger.metadata("accuracy", {"unit": "%", "GOAL": "MAXIMIZE", "STAGE": "VAL"})
json_logger.metadata("device", {"unit": "INFO", "GOAL": "NONE", "STAGE": "INFO"})
json_logger.metadata("epoch_start", {"unit": "INFO", "GOAL": "NONE", "STAGE": "TRAIN"})
json_logger.metadata("model_save", {"unit": "INFO", "GOAL": "NONE", "STAGE": "TRAIN"})
json_logger.metadata("training_completed", {"unit": "INFO", "GOAL": "NONE", "STAGE": "TRAIN"})


parser = argparse.ArgumentParser("Single_Path_One_Shot")
parser.add_argument('--exp_name', type=str, default='spos_c10_train_supernet', help='experiment name')
# Supernet Settings
parser.add_argument('--layers', type=int, default=20, help='batch size')
parser.add_argument('--num_choices', type=int, default=4, help='number choices per layer')
# Training Settings
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=600, help='batch size')
parser.add_argument('--num_steps', type=int, default=100, help='batch size')
parser.add_argument('--lr', type=float, default=0.025, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight-decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--print_freq', type=int, default=100, help='print frequency of training')
parser.add_argument('--val_interval', type=int, default=5, help='validate and save frequency')
parser.add_argument('--save_path', type=str, default='./checkpoints/', help='checkpoints direction')
parser.add_argument('--seed', type=int, default=0, help='training seed')
# Dataset Settings
parser.add_argument('--data_path', type=str, default='/data/teco-data/', help='dataset dir')
parser.add_argument('--classes', type=int, default=10, help='dataset classes')
parser.add_argument('--dataset', type=str, default='cifar10', help='path to the dataset')
parser.add_argument('--cutout', action='store_true', help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--auto_aug', action='store_true', default=False, help='use auto augmentation')
parser.add_argument('--resize', action='store_true', default=False, help='use resize')
args = parser.parse_args()
args.device = torch.device('sdaa') if torch.sdaa.is_available() else torch.device('cpu')
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logging.info(args)
utils.set_seed(args.seed)


def train(args, epoch, train_loader, model, criterion, optimizer):
    model.train()
    lr = optimizer.param_groups[0]["lr"]
    train_acc = utils.AverageMeter()
    train_loss = utils.AverageMeter()
    steps_per_epoch = len(train_loader)

    running_loss = 0.0
    correct = 0
    total = 0
    prev_loss = float('inf')
    iteration = 0
    for i, (inputs, targets) in enumerate(train_loader):
        start_time = time.time()
        
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        choice = utils.random_choice(args.num_choices, args.layers)
        
        optimizer.zero_grad()

        with torch.sdaa.amp.autocast():
            outputs = model(inputs, choice)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()    # loss缩放并反向转播
        scaler.step(optimizer)    # 参数更新
        scaler.update()    # 基于动态Loss Scale更新loss_scaling系数

        '''
        prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
        n = inputs.size(0)
        train_loss.update(loss.item(), n)
        train_acc.update(prec1.item(), n)

        
        # if step % args.print_freq == 0 or step == len(train_loader) - 1:
        logging.info(
            '[Supernet Training] lr: %.5f epoch: %03d/%03d, step: %03d/%03d, '
            'train_loss: %.3f(%.3f), train_acc: %.3f(%.3f)'
            % (lr, epoch+1, args.epochs, step+1, steps_per_epoch,
               loss.item(), train_loss.avg, prec1, train_acc.avg)
        )
        '''
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        batch_time = time.time() - start_time
        ips = args.batch_size / batch_time if batch_time > 0 else 0

        if loss.item() < prev_loss:
            json_logger.log(
                step=(epoch, iteration),
                data={
                    "rank": 0,
                    "train.loss": loss.item(),
                    "train.ips": ips
                },
                verbosity=Verbosity.DEFAULT
            )
            iteration += 1
            if iteration == args.num_steps:
                json_logger.log(
                    step=(epoch, iteration),
                    data={"stop_training": f"Epoch {epoch+1} reached {args.num_steps} iterations, stopping training"},
                    verbosity=Verbosity.DEFAULT
                )
                return

        prev_loss = loss.item()

        if iteration % args.num_steps == 0:
            avg_loss = running_loss / args.num_steps
            accuracy = args.num_steps * correct / total
            json_logger.log(
                step=(epoch, iteration),
                data={
                    "train_avg_loss": avg_loss,
                    "train_accuracy": accuracy
                },
                verbosity=Verbosity.DEFAULT
            )
            running_loss = 0.0
            correct = 0
            total = 0
            
    return 0, 0


def validate(args, val_loader, model, criterion):
    model.eval()
    val_loss = utils.AverageMeter()
    val_acc = utils.AverageMeter()
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            choice = utils.random_choice(args.num_choices, args.layers)
            outputs = model(inputs, choice)
            loss = criterion(outputs, targets)
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            val_loss.update(loss.item(), n)
            val_acc.update(prec1.item(), n)
    return val_loss.avg, val_acc.avg


def main():
    # Check Checkpoints Direction
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # Define Data
    assert args.dataset in ['cifar10', 'imagenet']
    train_transform, valid_transform = utils.data_transforms(args)
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='/data/teco-data/cifar10', train=True,
                                                download=False, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, num_workers=8)
        valset = torchvision.datasets.CIFAR10(root='/data/teco-data/cifar10', train=False,
                                              download=False, transform=valid_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=8)
    elif args.dataset == 'imagenet':
        train_data_set = datasets.ImageNet(os.path.join(args.data_path, args.dataset, 'train'), train_transform)
        val_data_set = datasets.ImageNet(os.path.join(args.data_path, args.dataset, 'valid'), valid_transform)
        train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=8, pin_memory=True, sampler=None)
        val_loader = torch.utils.data.DataLoader(val_data_set, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=8, pin_memory=True)
    else:
        raise ValueError('Undefined dataset !!!')

    # Define Supernet
    model = SinglePath_OneShot(args.dataset, args.resize, args.classes, args.layers)
    logging.info(model)
    model = model.to(args.device)
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, args.momentum, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    print('\n')

    # Running
    start = time.time()
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        # Supernet Training
        train_loss, train_acc = train(args, epoch, train_loader, model, criterion, optimizer)
        scheduler.step()
        '''
        logging.info(
            '[Supernet Training] epoch: %03d, train_loss: %.3f, train_acc: %.3f' %
            (epoch + 1, train_loss, train_acc)
        )
        # Supernet Validation
        val_loss, val_acc = validate(args, val_loader, model, criterion)
        # Save Best Supernet Weights
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_ckpt = os.path.join(args.save_path, '%s_%s' % (args.exp_name, 'best.pth'))
            torch.save(model.state_dict(), best_ckpt)
            logging.info('Save best checkpoints to %s' % best_ckpt)
        logging.info(
            '[Supernet Validation] epoch: %03d, val_loss: %.3f, val_acc: %.3f, best_acc: %.3f'
            % (epoch + 1, val_loss, val_acc, best_val_acc)
        )
        print('\n')
        '''

    # Record Time
    # utils.time_record(start)


if __name__ == '__main__':
    main()