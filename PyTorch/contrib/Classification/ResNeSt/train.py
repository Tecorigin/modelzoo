# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.
import argparse
import os
import sys
import time

import loguru
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from resnest import ResNeSt

# 加载模型
from utils import collate_fn, evaluate, get_datasets, train_one_epoch


def init_distributed_device(args):
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    distributed = local_rank != -1
    if distributed:
        if args.device == "cuda":
            if not torch.cuda.is_available():
                loguru.logger.error(f"CUDA is not available.")
                sys.exit(0)
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
        elif args.device == "sdaa":
            if not torch.sdaa.is_available():
                loguru.logger.error(f"SDAA is not available.")
                sys.exit(0)
            device = torch.device(f"sdaa:{local_rank}")
            torch.sdaa.set_device(device)
            torch.distributed.init_process_group(backend="tccl", init_method="env://")
        else:
            loguru.logger.error("This device type is not supported.")
            sys.exit(0)
    else:
        if args.device == "cuda":
            device = torch.device("cuda")
        elif args.device == "sdaa":
            device = torch.device("sdaa")
        elif args.device == "cpu":
            device = torch.device("cpu")
        else:
            loguru.logger.error("This device type is not supported.")
            sys.exit(0)

    args.local_rank = local_rank
    args.distributed = distributed
    return device


def main(args):
    device = init_distributed_device(args)

    dataset_path = args.dataset_path

    train_dataset, val_dataset = get_datasets(dataset_path)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    loguru.logger.info("Using {} dataloader workers every process".format(nw))

    if args.local_rank != -1:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    loguru.logger.info(f"创建 DataLoader")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        pin_memory=True,
        num_workers=nw,
        sampler=train_sampler,
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        pin_memory=True,
        num_workers=nw,
        sampler=val_sampler,
        collate_fn=collate_fn,
    )

    # 如果存在预训练权重则载入
    loguru.logger.info(f"创建模型并加载预训练权重")
    loguru.logger.info(f"将使用 {args.model_name}")
    model = ResNeSt.create_pretrained(args.model_name)
    if "sdaa" in args.device and not args.amp:
        loguru.logger.warning(f"sdaa 训练模式下推荐开启混合精度训练")
    if args.precision == "float16":
        args.dtype = torch.float16
    elif args.precision == "float32":
        args.dtype = torch.float32
    elif args.precision == "bfloat16":
        args.dtype = torch.bfloat16
    else:
        loguru.logger.error(f"使用了暂不支持的 precision ({args.precision})")
        sys.exit(-1)

    model.to(device, dtype=args.dtype)

    if args.pretrained_path is not None:
        pretrained_path = args.pretrained_path
        assert os.path.exists(pretrained_path), f"weights file: {pretrained_path} not exist."
        weights_dict = torch.load(args.weights, map_location="cpu")
        weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
        loguru.logger.info(model.load_state_dict(weights_dict, strict=False))

    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1e-4)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.01 * args.lrf)
    if "sdaa" in args.device:
        scaler = torch.sdaa.amp.GradScaler()
    else:
        scaler = None

    best_acc = 0.0
    global_step = 0

    val_losses = []
    val_accuracies = []

    loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    loss_function.to(device)

    loguru.logger.info(f"开始训练")
    for epoch in range(args.epochs):
        if args.local_rank != -1:
            train_sampler.set_epoch(epoch)
        # 记录训练时间
        start_time = time.time()
        train_one_epoch(
            model=model,
            loss_function=loss_function,
            optimizer=optimizer,
            scaler=scaler,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            args=args,
        )
        scheduler.step()

        end_time = time.time()
        train_time = end_time - start_time

        loguru.logger.info(f"第 {epoch} 训练共花费: {train_time:.3f} seconds")

        if args.max_step < 0:
            val_loss, val_acc = evaluate(model=model, data_loader=val_loader, device=device, epoch=epoch, args=args)
        else:
            break
        global_step += 1

        if args.local_rank == 0:
            best_model_name = f"best_model_batchsize{batch_size}_lr{args.str_lr}.pth"
            latest_model_name = f"latest_model_batchsize{batch_size}_lr{args.str_lr}.pth"

            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            if args.distributed:
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.module.state_dict(), os.path.join(args.path, best_model_name))
                torch.save(model.module.state_dict(), os.path.join(args.path, latest_model_name))
            else:
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), os.path.join(args.path, best_model_name))
                torch.save(model.state_dict(), os.path.join(args.path, latest_model_name))


if __name__ == "__main__":
    # 设置IP:PORT，框架启动TCP Store为ProcessGroup服务
    os.environ["MASTER_ADDR"] = "localhost"  # 设置IP

    parser = argparse.ArgumentParser()

    # 我认为有用的参数 (一部分是原来就有的, 另一部分是从 timm 上拿过来的)
    parser.add_argument("--epochs", type=int, default=300, metavar="N", help="number of epochs to train (default: 300)")
    parser.add_argument("--batch_size", type=int, default=128, help="Input batch size for training (default: 128)")
    parser.add_argument(
        "--precision", default="float32", type=str, help="Numeric precision. One of (float32, float16, bfloat16)"
    )
    parser.add_argument("--device", default="sdaa", type=str, help="Device (accelerator) to use.")
    parser.add_argument("--max_step", default=-1, type=int)

    # 路径问题
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument(
        "--pretrained_path",
        default=None,
        type=str,
        help="Load this checkpoint as if they were the pretrained weights (with adaptation).",
    )
    parser.add_argument("--save_path", type=str, default="", help="Path to save checkpoints.")

    # 优化器参数
    parser.add_argument("--lr", type=float, default=0.1, metavar="LR", help="learning rate (default: 0.1)")
    parser.add_argument("--lrf", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay (default: 1e-4)")
    parser.add_argument("--momentum", type=float, default=0.9, help="Optimizer momentum (default: 0.9)")
    parser.add_argument(
        "--amp", action="store_true", default=False, help="use NVIDIA Apex AMP or Native AMP for mixed precision training"
    )
    parser.add_argument("--model_name", type=str, default="resnest50", help="Name of the model.")

    opt = parser.parse_args()

    opt.str_lr = str(opt.lr).replace(".", "_")
    opt.save_path = os.path.join(opt.save_path, f"batchsize{opt.batch_size}_lr{opt.str_lr}")
    os.makedirs(opt.save_path, exist_ok=True)

    main(opt)
