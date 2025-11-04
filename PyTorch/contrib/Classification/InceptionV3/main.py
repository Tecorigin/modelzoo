import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch_sdaa
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, seed_everything

from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def setup_distributed(local_rank):
    device = torch.device(f"sdaa:{local_rank}")
    torch.sdaa.set_device(device)
    torch.distributed.init_process_group(backend="tccl", init_method="env://")
    return device

def train(epoch, scaler, global_step):
    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(cifar10_training_loader):

        labels = labels.to(device)
        images = images.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        n_iter = (epoch - 1) * len(cifar10_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        logger.log(
            step=(epoch, global_step),
            data={"loss": loss.item(), "speed": len(images) / (time.time() - start)},
            verbosity=Verbosity.DEFAULT,
        )

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar10_training_loader.dataset)
        ))

        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

        global_step += 1

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    return global_step

@torch.no_grad()
def eval_training(epoch=0, tb=True):
    start = time.time()
    net.eval()

    test_loss = 0.0
    correct = 0.0

    for (images, labels) in cifar10_test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar10_test_loader.dataset),
        correct.float() / len(cifar10_test_loader.dataset),
        finish - start
    ))
    print()

    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar10_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(cifar10_test_loader.dataset), epoch)

    logger.log(
        step=(epoch,),
        data={"val.loss": test_loss / len(cifar10_test_loader.dataset), "val.accuracy": correct.float() / len(cifar10_test_loader.dataset)},
        verbosity=Verbosity.DEFAULT,
    )

    return correct.float() / len(cifar10_test_loader.dataset)

if __name__ == '__main__':
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument("-device", default='sdaa', type=str, choices=['cpu', 'cuda', 'sdaa'], help="which device to use, sdaa default")
    parser.add_argument('-b', type=int, default=512, help='batch size for dataloader')
    parser.add_argument('-epoch', type=int, default=50, help='training epoch')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-dataset_path', default="./dataset/cifar10", type=str, help='path to dataset')
    parser.add_argument("-distributed", action='store_true', help="Whether to run training.")
    parser.add_argument('-nproc_per_node', default=4, type=int, help="The number of processes to launch on each node")
    parser.add_argument("-amp", default=False, action='store_true', help="Mixed precision training")
    args = parser.parse_args()

    seed_everything(2024)

    if args.distributed:
        device = setup_distributed(local_rank)
    else:
        device = torch.device('sdaa' if torch.sdaa.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    settings.EPOCH = args.epoch
    fp16 = args.amp
    world_size = args.nproc_per_node
    batch_size = args.b
    dataset_path = args.dataset_path

    if fp16:
        scaler = torch_sdaa.amp.GradScaler()
    else:
        scaler = None
    print("using device: ", device)

    net = get_network(args)
    net.to(device)
    if args.distributed:
        net = DDP(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(settings.CIFAR10_TRAIN_MEAN, settings.CIFAR10_TRAIN_STD)
    ])
    cifar10_training = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform_train)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(settings.CIFAR10_TRAIN_MEAN, settings.CIFAR10_TRAIN_STD)
    ])
    cifar10_test = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform_test)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(cifar10_training, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(cifar10_test, shuffle=False)
        batch_size = batch_size // world_size
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    cifar10_training_loader = get_training_dataloader(
        settings.CIFAR10_TRAIN_MEAN,
        settings.CIFAR10_TRAIN_STD,
        num_workers=0,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        dataset_path=dataset_path
    )

    cifar10_test_loader = get_test_dataloader(
        settings.CIFAR10_TRAIN_MEAN,
        settings.CIFAR10_TRAIN_STD,
        num_workers=0,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        dataset_path=dataset_path
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2)
    iter_per_epoch = len(cifar10_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    logger = Logger(
        [
            StdOutBackend(Verbosity.DEFAULT),
            JSONStreamBackend(Verbosity.VERBOSE, "train_log1.json"),
        ]
    )
    logger.metadata("loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
    logger.metadata("speed", {"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})
    logger.metadata("val.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "VALID"})
    logger.metadata("val.accuracy", {"unit": "", "GOAL": "MAXIMIZE", "STAGE": "VALID"})

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    writer = SummaryWriter(log_dir=os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW))

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))

    global_step = 0
    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        global_step = train(epoch, scaler, global_step)
        acc = eval_training(epoch)

        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    writer.close()
