import os
import sys
import time
import copy
import argparse

import numpy as np
from timm.models.helpers import load_checkpoint
from mobilenet import mobilenet
from resnet import resnet34

import torch
import torch_sdaa
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision.utils
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity

try:
    from torchprofile import profile_macs
except ImportError:
    print("to calculate flops, get torchprofile from https://github.com/mit-han-lab/torchprofile")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.backends.cudnn.benchmark = True
local_rank = int(os.environ.get("LOCAL_RANK", -1))

def setup_distributed(local_rank):
    device = torch.device(f"sdaa:{local_rank}")
    torch.sdaa.set_device(device)
    torch.distributed.init_process_group(backend="tccl", init_method="env://")
    return device

parser = argparse.ArgumentParser(description='PyTorch Transfer to CIFAR Training')
parser.add_argument('--seed', type=int, default=12, help='random seed')
parser.add_argument('--model_name', type=str, help='model name')
parser.add_argument('--grad_scale', type=str, help='grad scale')
parser.add_argument('--autocast', type=str, help='autocast')
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar100', help='cifar10, cifar100, or cinic10')
parser.add_argument('--batch_size', type=int, default=48, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for data loading')
parser.add_argument('--n_gpus', type=int, default=1, help='number of available gpus for training')
parser.add_argument('--lr', type=float, default=0.01, help='init learning rate')
parser.add_argument('--drop', type=float, default=0.2, help='drop out rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--autoaugment', action='store_true', default=False, help='use auto augmentation')
parser.add_argument('--save', action='store_true', default="./pt", help='dump output')
# model related
parser.add_argument('--model', type=str, default=None,
                    help='location of a json file of specific model declaration')
parser.add_argument('--imagenet', type=str, default=None,
                    help='location of initial weight to load')
                    
parser.add_argument("-distributed", action='store_true', help="Whether to run training.")

parser.add_argument('-nproc_per_node', default=3, type=int,
                    help="The number of processes to launch on each node, "
                    "for GPU training, this is recommended to be set "
                    "to the number of GPUs in your system so that "
                    "each process can be bound to a single GPU.")
args = parser.parse_args()


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
dataset = args.dataset

# Initialize TCAP_DLLogger
logger = Logger(
    [
        StdOutBackend(Verbosity.DEFAULT),
        JSONStreamBackend(Verbosity.VERBOSE, "log.json"),
    ]
)

if args.save:
    args.save = 'mobilenet-{}-{}'.format(dataset, time.strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)
    print('Experiment dir : {}'.format(args.save))

if args.distributed:
    device = setup_distributed(local_rank)
else:
    device = torch.device('sdaa' if torch.sdaa.is_available() else 'cpu')
    local_rank = 0
    rank = 0

NUM_CLASSES = 100 if 'cifar100' or 'cifar-fs' in dataset else 10

if args.autoaugment:
    try:
        from autoaugment import CIFAR10Policy
    except ImportError:
        print("cannot import autoaugment, setting autoaugment=False")
        print("autoaugment is available "
              "from https://github.com/DeepVoltaire/AutoAugment")
        args.autoaugment = False



def main():
    # if not torch.cuda.is_available():
        # logger.info(data='no gpu device available')
        # sys.exit(1)

    logger.info(data={"args": vars(args)})

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    best_acc = 0  # initiate a artificial best accuracy so far
    top_checkpoints = []  # initiate a list to keep track of

    # Data
    train_transform, valid_transform = _data_transforms(args)
    if dataset == 'cifar100':
        train_data = torchvision.datasets.CIFAR100(
            root=args.data, train=True, download=True, transform=train_transform)
        valid_data = torchvision.datasets.CIFAR100(
            root=args.data, train=False, download=True, transform=valid_transform)
    elif dataset == 'cifar10':
        train_data = torchvision.datasets.CIFAR10(
            root=args.data, train=True, download=True, transform=train_transform)
        valid_data = torchvision.datasets.CIFAR10(
            root=args.data, train=False, download=True, transform=valid_transform)
    else:
        raise KeyError
    world_size = args.nproc_per_node
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(valid_data , shuffle=False)
        batch_size = args.batch_size // world_size
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True
        batch_size = args.batch_size

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=(train_sampler is None), pin_memory=True, num_workers=args.num_workers, sampler=train_sampler)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers, sampler=val_sampler)
    
    
    
    
    net = mobilenet(num_classes=NUM_CLASSES)  # assuming transfer from ImageNet

    #load_checkpoint(net, args.imagenet, use_ema=True)

    #net.reset_classifier(num_classes=NUM_CLASSES)
    #net.drop_rate = args.drop

    # calculate #Paramaters and #FLOPS
    params = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
    try:
        inputs = torch.randn(1, 3, 224, 224)
        flops = profile_macs(copy.deepcopy(net), inputs) / 1e6
        logger.info(data={'#params': params, '#flops': flops})
    except:
        logger.info(data={'#params': params})

    if args.n_gpus > 1:
        #net = nn.DataParallel(net)  # data parallel in case more than 1 gpu available
        params = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6

    net = net.to(device)
    if args.distributed:
        net = DDP(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    n_epochs = args.epochs

    parameters = filter(lambda p: p.requires_grad, net.parameters())

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD(parameters,
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    scaler = GradScaler() 

    for epoch in range(n_epochs):

        logger.info(data={'epoch': epoch, 'lr': scheduler.get_lr()[0]})

        train(train_queue, net, criterion, optimizer, scaler, epoch,scheduler)  
        _, valid_acc = infer(valid_queue, net, criterion, epoch)

        # checkpoint saving
        if args.save:
            if valid_acc > best_acc:
                torch.save(net.state_dict(), os.path.join(args.save, 'weights.pt'))
                best_acc = valid_acc

        scheduler.step()
        
        
def train(train_queue, net, criterion, optimizer, scaler, epoch, scheduler):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    
    rank = -1  # Single GPU, non-DDP case
    if dist.is_initialized():
        rank = dist.get_rank()

    for step, (inputs, targets) in enumerate(train_queue):
        # upsample by bicubic to match imagenet training size
        inputs = F.interpolate(inputs, size=224, mode='bicubic', align_corners=False)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        with autocast():
            outputs = net(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        elapsed_time = time.time() - start_time
        throughput = total / elapsed_time

        if step % args.report_freq == 0:
            logger.log(
                step=(epoch, step),
                data={
                "rank": rank,
                "train_loss": train_loss / total,
                "train_acc": 100. * correct / total,
                "train.ips": throughput,
                "data.shape": str(inputs.shape),
                "train.lr": scheduler.get_lr()[0]
                },
                verbosity=Verbosity.DEFAULT,
            )

    elapsed_time = time.time() - start_time
    throughput = total / elapsed_time
    logger.log(
        step=(epoch, step),
        data={
        "rank": rank,
        "train_epoch_loss": train_loss / total,
        "train_epoch_acc": 100. * correct / total,
        "train.ips": throughput,
        "train.lr": scheduler.get_lr()[0]
        },
        verbosity=Verbosity.DEFAULT,
    )

    return train_loss/total, 100.*correct/total


def infer(valid_queue, net, criterion, epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    
    rank = -1  # Single GPU, non-DDP case
    if dist.is_initialized():
        rank = dist.get_rank()

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_queue):
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast():
                outputs = net(inputs)
                loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            elapsed_time = time.time() - start_time
            throughput = total / elapsed_time

            if step % args.report_freq == 0:
                logger.log(
                    step=(epoch, step),
                    data={
                    "rank": rank,
                    "valid_loss": test_loss / total,
                    "valid_acc": 100. * correct / total,
                    "valid.ips": throughput,
                    "data.shape": str(inputs.shape)
                    },
                    verbosity=Verbosity.DEFAULT,
                )

    acc = 100.*correct/total
    elapsed_time = time.time() - start_time
    throughput = total / elapsed_time
    logger.log(
        step=(epoch, step),
        data={
        "rank": rank,
        "valid_epoch_loss": test_loss / total,
        "valid_epoch_acc": 100. * correct / total,
        "valid.ips": throughput,
        },
        verbosity=Verbosity.DEFAULT,
    )

    return test_loss/total, acc


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms(args):

    if 'cifar' in args.dataset:
        norm_mean = [0.49139968, 0.48215827, 0.44653124]
        norm_std = [0.24703233, 0.24348505, 0.26158768]
    else:
        raise KeyError

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.Resize(224, interpolation=3),  # BICUBIC interpolation
        transforms.RandomHorizontalFlip(),
    ])

    if args.autoaugment:
        train_transform.transforms.append(CIFAR10Policy())

    train_transform.transforms.append(transforms.ToTensor())

    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    train_transform.transforms.append(transforms.Normalize(norm_mean, norm_std))

    valid_transform = transforms.Compose([
        transforms.Resize(224, interpolation=3),  # BICUBIC interpolation
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    return train_transform, valid_transform


if __name__ == '__main__':
    main()
