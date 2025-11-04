import os
import argparse
import time
import math
from argparse import ArgumentParser,ArgumentTypeError

import torch
import torch_sdaa
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms

import random
from my_dataset import MyDataSet
import numpy as np

from utils import train_one_epoch, evaluate
import torch.nn as nn
from model import se_resnet50 as create_model
# 导入DDP所需的依赖库
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
import matplotlib.pyplot as plt
from pathlib import Path
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
    # DDP backend初始化
    else:
        device = torch.device(f"sdaa:{local_rank}")
        torch.sdaa.set_device(device)
        # 初始化ProcessGroup，通信后端选择tccl
        torch.distributed.init_process_group(backend="tccl", init_method="env://")

    print(args)

    data_root = args.dataset_path
    img_size = 244      

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(root_dir=data_root,
                              txt_name='train_list.txt',
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(root_dir=data_root,
                            txt_name='val_list.txt',
                            transform=data_transform["val"])

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
                                               num_workers=nw,
                                               sampler=train_sampler,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             sampler=val_sampler,
                                             collate_fn=val_dataset.collate_fn)

    # 如果存在预训练权重则载入
    model = create_model(num_classes=args.num_classes).to(device)

    if args.distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

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
        # 记录训练时间
        start_time = time.time()
        train_throughput = len(train_loader.dataset)  # 计算训练吞吐量
        train_loss, train_acc, train_data_to_device_time, train_compute_time, total_forward_time, total_backward_time, total_optimizer_step_time = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                scaler=scaler,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                use_acm = args.autocast,
                                                rank = opt.local_rank,
                                                json_logger=json_logger,
                                                img_size=img_size,
                                                lr=args.lr,
                                                train_throughput=train_throughput,
                                                max_step=args.step)
        scheduler.step()
        
        end_time = time.time()
        train_time = end_time - start_time

        # json_logger.log(
        #     step = (epoch, global_step),
        #     data = {
        #             "rank":args.local_rank,
        #             "train.loss":train_loss,
        #             "train.ips":train_throughput,
        #             "data.shape":[img_size, img_size],
        #             "train.lr":args.lr,
        #             "train.data_time":train_data_to_device_time,
        #             "train.compute_time":train_compute_time,
        #             "train.fp_time":total_forward_time,
        #             "train.bp_time":total_backward_time,
        #             "train.grad_time":total_optimizer_step_time,
        #             },
        #     verbosity=Verbosity.DEFAULT,)

        if args.step < 0: 
            val_loss, val_acc = evaluate(model=model,
                                        data_loader=val_loader,
                                        device=device,
                                        epoch=epoch)
            json_logger.log(
                step = (epoch, global_step),
                data = {
                        "rank":args.local_rank,
                        "val.acc":val_acc,
                        "val.loss":val_loss,
                        },
                verbosity=Verbosity.DEFAULT,)
        else :
            break
        global_step += 1
        
        if args.local_rank == 0:
            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]


            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            if args.distributed:
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.module.state_dict(), "./weights/best_model.pth")
                torch.save(model.module.state_dict(), "./weights/latest_model.pth")
            else:
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), "./weights/best_model.pth")
                torch.save(model.state_dict(), "./weights/latest_model.pth")

    if args.local_rank == 0 and args.step < 0:

        plt.figure()
        plt.plot(range(args.epochs), train_losses, label='Train Loss')
        plt.plot(range(args.epochs), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curve')
        plt.savefig('./experiments/loss_curve.png')
        
        # 画出精度曲线
        plt.figure()
        plt.plot(range(args.epochs), train_accuracies, label='Train Accuracy')
        plt.plot(range(args.epochs), val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy Curve')
        plt.savefig('./experiments/accuracy_curve.png')
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnodes", default=1, type=int)
    parser.add_argument("--local-rank", default= -1, type=int)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--distributed', type=str2bool, default=True)
    parser.add_argument('--autocast', type=str2bool, default=True)
    parser.add_argument("--step", default=-1, type=int)
    parser.add_argument('--dataset_path', type=str,
                        default="/mnt/nvme/common/train_dataset/mini-imagenet")
    parser.add_argument('--model_name', type=str,
                    default="senet")
    parser.add_argument('--device', default='sdaa')
    parser.add_argument('--path', type=str, default='./experiments/')

    opt = parser.parse_args()
    directory = Path(opt.path).parent
    if not directory.exists():
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    json_logger = Logger(
        [
            StdOutBackend(Verbosity.DEFAULT),
            JSONStreamBackend(Verbosity.VERBOSE, opt.path+"log.json"),
        ]
    )
    if opt.local_rank == 0:
        json_logger.info(data=opt)
        json_logger.info(data="start training ...")
    local_rank = opt.local_rank

    main(opt)
