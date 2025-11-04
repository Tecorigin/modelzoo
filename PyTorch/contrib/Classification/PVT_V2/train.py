import os
import argparse
import time
import math
from argparse import ArgumentParser, ArgumentTypeError
# Adapted to tecorigin hardware
import torch
import torch_sdaa
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import random
import numpy as np
from utils import train_one_epoch, evaluate, collate_fn, get_datasets
import torch.nn as nn
# 导入DDP所需的依赖库
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
import matplotlib.pyplot as plt

# 加载模型
from pvt_v2 import PyramidVisionTransformerV2

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


    dataset_path = args.dataset_path
    img_size = 224        

    train_dataset, val_dataset = get_datasets(dataset_path)


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
                                               collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=(train_sampler is None),
                                             pin_memory=True,
                                             num_workers=nw,
                                             sampler=val_sampler,
                                             collate_fn=collate_fn)

    # 如果存在预训练权重则载入
    model = PyramidVisionTransformerV2()
    model.to(args.device)
    if args.weights != "":
        if args.weights is None:
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load(args.weights, map_location=device)
            weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
            # 删除有关分类类别的权重
            for k in list(weights_dict.keys()):
                if "classifier" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))
        elif os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))


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
                step=(epoch, global_step),
                data={
                    "rank": args.local_rank,
                    "val.acc": val_acc,
                    "val.loss": val_loss,
                },
                verbosity=Verbosity.DEFAULT, )
        else:
            break
        global_step += 1

        if args.local_rank == 0:
            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            
            best_model_name = f'best_model_batchsize_{args.batch_size}_lr_{args.str_lr}.pth'
            latest_model_name = f'latest_model_batchsize_{args.batch_size}_lr_{args.str_lr}.pth'
            
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
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

    if args.local_rank == 0 and args.step < 0:
        # 获取scripts文件夹路径
        scripts_path = os.path.join(os.path.dirname(os.path.dirname(args.path)), 'scripts')

        plt.figure()
        plt.plot(range(args.epochs), train_losses, label='Train Loss')
        plt.plot(range(args.epochs), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curve')
        #plt.savefig(os.path.join(args.path, f'loss_curve_batch_size_{args.batch_size}_lr_{args.str_lr}.png'))
        plt.savefig(os.path.join(scripts_path, 'train_sdaa_3rd_loss.png'))
        # 画出精度曲线
        plt.figure()
        plt.plot(range(args.epochs), train_accuracies, label='Train Accuracy')
        plt.plot(range(args.epochs), val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy Curve')
        #plt.savefig(os.path.join(args.path, f'accuracy_curve_batch_size_{args.batch_size}_lr_{args.str_lr}.png'))
        plt.savefig(os.path.join(scripts_path, 'train_sdaa_3rd_acc.png'))
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnodes", default=1, type=int)
    parser.add_argument("--local-rank", default= -1, type=int)
    parser.add_argument('--num_classes', type=int, default=1000)
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
                    default="efficientnet")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='None',
                        help='initial weights path')
    parser.add_argument('--freeze_layers', type=bool, default=False)
    parser.add_argument('--device', default='sdaa')
    parser.add_argument('--path', type=str, default='./experiments/')
    parser.add_argument('--weights_path', type=str, default='weights')

    opt = parser.parse_args()
    
    opt.str_lr = str(opt.lr).replace('.', '_')
    if not opt.distributed:
        opt.path = './single_experiments/'
    opt.path = os.path.join(opt.path, "train_sdaa_3rd.log")
    os.makedirs(opt.path, exist_ok=True)
    
    
    json_logger = Logger(
        [
            StdOutBackend(Verbosity.DEFAULT),
            # JSONStreamBackend(Verbosity.VERBOSE, opt.path+"log.json"),
            JSONStreamBackend(Verbosity.VERBOSE, os.path.join(opt.path, "train_sdaa_3rd.log")),
        ]
    )
    json_logger.metadata("train.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
    json_logger.metadata("train.loss_mean", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
    json_logger.metadata("val.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "VALID"})
    json_logger.metadata("train.ips", {"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})
    json_logger.metadata("val.ips", {"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "VALID"})
    json_logger.metadata("train.compute_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
    json_logger.metadata("train.fp_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
    json_logger.metadata("train.bp_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
    json_logger.metadata("train.grad_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
    if opt.local_rank == 0:
        json_logger.info(data=opt)
        json_logger.info(data="start training ...")
    local_rank = opt.local_rank

    main(opt)
