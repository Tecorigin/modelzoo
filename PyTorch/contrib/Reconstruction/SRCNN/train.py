# python -m torch.distributed.launch --nproc_per_node 3 --master_port=25641 train.py  --train_file "BLAH_BLAH/91-image_x3.h5" --eval_file "BLAH_BLAH/Set5_x3.h5" --outputs_dir "BLAH_BLAH/outputs" --scale 3 --lr 1e-4 --batch_size 512 --num_epochs 20 --num_workers 8 --seed 123 --use_amp True --use_ddp True

import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import SRCNN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr

import torch_sdaa
import time
from torch.nn.parallel import DistributedDataParallel as DDP


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


# 初始化logger
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
json_logger = Logger(
    [
        StdOutBackend(Verbosity.DEFAULT),
        JSONStreamBackend(Verbosity.VERBOSE, 'dlloger_example.json'),
    ]
)

json_logger.metadata("train.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
# json_logger.metadata("train.loss_mean", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
# json_logger.metadata("val.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "VALID"})
json_logger.metadata("train.ips",{"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("data.shape", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.lr", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
# json_logger.metadata("val.ips",{"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "VALID"})
json_logger.metadata("train.data_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.compute_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.fp_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.bp_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.grad_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})

def format_value(value):
    return "{:.3f}".format(value)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc_per_node',type=int,default=1)
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--eval_file', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', required=False, default='sdaa', type=str,
                        help='which device to use. cuda, sdaa optional, sdaa default')
    parser.add_argument("--use_amp", required=False, default=False, type=str2bool, help='Distributed training or not')
    parser.add_argument("--use_ddp", required=False, default=False, type=str2bool, help='DDP training or not')
    parser.add_argument("--local-rank", required=False, default=-1, type=int)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    use_ddp = args.use_ddp

    if use_ddp:
        print('use ddp')
        local_rank = args.local_rank
        # DDP backend初始化
        device = torch.device(f"sdaa:{local_rank}")
        torch.sdaa.set_device(device)
        # 初始化ProcessGroup，通信后端选择tccl
        torch.distributed.init_process_group(backend="tccl", init_method="env://")
    else:
        local_rank = 0
        device = args.device

    use_amp = args.use_amp

    torch.manual_seed(args.seed)

    model = SRCNN().to(device)
    if use_ddp:
        model = DDP(model)
    else:
        model = model

    if use_amp:
        print('use amp')
        scaler = torch_sdaa.amp.GradScaler()  # 定义GradScaler
    else:
        scaler = None
    
    criterion = nn.MSELoss()
    if use_ddp:
        optimizer = optim.Adam([
        {'params': model.module.conv1.parameters()},
        {'params': model.module.conv2.parameters()},
        {'params': model.module.conv3.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)
    else:
        optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    #log============
    global_step = 0
    # 创建 CUDA 事件
    start_event = torch.sdaa.Event(enable_timing=True)
    end_event = torch.sdaa.Event(enable_timing=True)
    start_step_time = time.time()
    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        if local_rank == 0:
            t = tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size))
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

        # with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
        #     t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

        for data in train_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            start_data_time = time.time()# 记录数据加载开始时间
            start_event.record()# 记录前向传播开始时间

            preds = model(inputs)

            end_event.record()# 记录前向传播结束时间
            torch.sdaa.synchronize()  # 等待 GPU 操作完成
            fp_time = start_event.elapsed_time(end_event) / 1000.0  # 计算前向传播时间
            start_compute_time = time.time()# 记录计算时间开始时间

            if use_amp:
                with torch_sdaa.amp.autocast():   # 开启AMP环境
                    preds = model(inputs)    # 前向计算
                    loss = criterion(preds, labels)    # 损失函数计算
                epoch_losses.update(loss.item(), len(inputs))
                optimizer.zero_grad()
                scaler.scale(loss).backward()    # loss缩放并反向转播
                scaler.step(optimizer)    # 参数更新
                scaler.update()    # 基于动态Loss Scale更新loss_scaling系数
            else:
                preds = model(inputs)
                loss = criterion(preds, labels)
                epoch_losses.update(loss.item(), len(inputs))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            end_compute_time = time.time() # 记录计算时间结束时间
            data_time = start_compute_time - start_data_time # 计算数据加载时间
            compute_time = end_compute_time - start_compute_time # 计算计算时间
            bp_time = time.time() - end_compute_time # 计算反向传播时间
            grad_time = time.time() - end_compute_time # 计算梯度更新时间

            if local_rank == 0:
                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
                # 计算每个训练步骤的 IPS
                end_step_time = time.time()                
                time_per_step = end_step_time - start_step_time
                ips = 1 / time_per_step  # 计算每秒完成的迭代次数
                start_step_time = end_step_time  # 更新下一个步骤的开始时间

                # 在这里添加你的日志记录代码
                json_logger.log(
                    step=(epoch, global_step),
                    data={
                        "rank": os.environ.get("LOCAL_RANK", "0"),
                        "train.loss": format_value(loss.item()),
                        "train.ips": format_value(ips),
                        "data.shape": list(inputs.shape),
                        "train.lr": optimizer.param_groups[0]['lr'],
                        "train.data_time": format_value(data_time),
                        "train.compute_time": format_value(compute_time),
                        "train.fp_time": format_value(fp_time),
                        "train.bp_time": format_value(bp_time),
                        "train.grad_time": format_value(grad_time),
                    },
                    verbosity=Verbosity.DEFAULT,
                )
                global_step += 1
                torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        # torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
