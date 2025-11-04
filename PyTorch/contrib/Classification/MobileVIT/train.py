import os
import argparse
import time
import math
from argparse import ArgumentParser,ArgumentTypeError

import torch
import torch_sdaa
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from model import mobile_vit_xx_small as create_model
from utils import read_split_data, train_one_epoch, evaluate
import torch.nn as nn
# 导入DDP所需的依赖库
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
import matplotlib.pyplot as plt
from pathlib import Path

# json_logger.metadata("train.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
# json_logger.metadata("train.loss_mean", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
# json_logger.metadata("val.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "VALID"})
# json_logger.metadata("train.ips",{"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})
# json_logger.metadata("val.ips",{"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "VALID"})
# json_logger.metadata("train.compute_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
# json_logger.metadata("train.fp_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
# json_logger.metadata("train.bp_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
# json_logger.metadata("train.grad_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
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
def main(args):
    print(args.distributed)
    if args.model_name != "MobileVit":
        raise Exception("Sorry, the model you choose is not supported now")

    if args.distributed is False:
        device = torch.device(args.device)
    # DDP backend初始化
    else:
        device = torch.device(f"sdaa:{local_rank}")
        torch.sdaa.set_device(device)
        # 初始化ProcessGroup，通信后端选择tccl
        torch.distributed.init_process_group(backend="tccl", init_method="env://")
        # device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    if local_rank != -1:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    # print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=(train_sampler is None),
                                               pin_memory=True,
                                               num_workers=0,
                                               sampler=train_sampler,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0,
                                             sampler=val_sampler,
                                             collate_fn=val_dataset.collate_fn)
    model = create_model(num_classes=args.num_classes).to(device)

    if args.weights != "" :
        if args.weights is None:
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load(args.weights, map_location=device)
            weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
            # 删除有关分类类别的权重
            for k in list(weights_dict.keys()):
                if "classifier" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "classifier" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
    if args.distributed:
        # ddp_model = create_model(num_classes=args.num_classes).to(device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0.005)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - 0.1) + 0.1  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scaler = torch_sdaa.amp.GradScaler()

    best_acc = 0.
    golbal_step = 0
    
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
                                                use_acm = args.use_amp,
                                                rank = opt.local_rank,
                                                json_logger=json_logger,
                                                img_size=img_size,
                                                lr=args.lr,
                                                train_throughput=train_throughput)
        scheduler.step()
        
        end_time = time.time()
        train_time = end_time - start_time

        
        # if opt.local_rank == 0:
        #     json_logger.log(
        #         step = (epoch, golbal_step),
        #         data = {
        #                 # "rank":args.local_rank,
        #                 "train.loss":train_loss, 
        #                 "train.ips":train_throughput,
        #                 "data.shape":[img_size, img_size],
        #                 "train.lr":args.lr,
        #                 "train.data_time":train_data_to_device_time,
        #                 "train.compute_time":train_compute_time,
        #                 "train.fp_time":total_forward_time,
        #                 "train.bp_time":total_backward_time,
        #                 "train.grad_time":total_optimizer_step_time,
        #                 },
        #         verbosity=Verbosity.DEFAULT,)
        #     golbal_step += 1

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        if args.local_rank == 0:
            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "./weights/best_model.pth")

            torch.save(model.state_dict(), "./weights/latest_model.pth")
         # 画出损失曲线
    if args.local_rank == 0:
        plt.figure()
        plt.plot(range(args.epochs), train_losses, label='Train Loss')
        # plt.plot(range(args.epochs), val_losses, label='Validation Loss')
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
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument('--num_classes', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.005)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="./data/UCM")
    parser.add_argument('--model-name', type=str,
                    default="MobileVit")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./mobilevit_xxs.pt',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=str2bool, default=False)
    parser.add_argument('--device', default='sdaa')
    parser.add_argument('--distributed', type=str2bool, default=False)
    parser.add_argument('--use_amp', type=str2bool, default=True)
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
    # logger.log(
    #             step = global_step,
    #             data = {"loss":_CE_loss.item(), "speed":ips},
    #             verbosity=Verbosity.DEFAULT,
    #         )

    main(opt)
