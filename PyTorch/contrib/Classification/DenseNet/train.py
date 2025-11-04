import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import datetime
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
from torch.sdaa import amp
from opt import get_args
from dataloader import get_data_loaders
from densenet import DenseNet

scaler = torch.sdaa.amp.GradScaler()    # 定义GradScaler

# 初始化 JSON Logger
json_logger = Logger(
    [
        StdOutBackend(Verbosity.DEFAULT),
        JSONStreamBackend(Verbosity.VERBOSE, "tmp.json"),
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

def setup_device():
    """设置设备（SDAA或CPU）"""
    device = torch.device('sdaa' if torch.sdaa.is_available() else 'cpu')
    json_logger.log(
        step=(),
        data={"device": f"Using device: {device}"},
        verbosity=Verbosity.DEFAULT
    )
    return device

def setup_model_and_optimizer(device, args):
    """初始化模型、损失函数、优化器和调度器"""
    model = DenseNet(args).to(device)  # 传递 args 作为 opt
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    return model, criterion, optimizer, scheduler

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    prev_loss = float('inf')
    iteration = 0

    for i, (images, labels) in enumerate(train_loader):
        start_time = time.time()
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()

        with torch.sdaa.amp.autocast():   # 开启AMP环境
            outputs = model(images)    
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()    # loss缩放并反向传播
        scaler.step(optimizer)    # 参数更新
        scaler.update()    # 基于动态Loss Scale更新loss_scaling系数
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
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
            if iteration > args.num_steps:
                json_logger.log(
                    step=(epoch, iteration),
                    data={"stop_training": f"Epoch {epoch+1} reached {args.num_steps} iterations, stopping training"},
                    verbosity=Verbosity.DEFAULT
                )
                return

        prev_loss = loss.item()

        if iteration % args.num_steps == 0:
            avg_loss = running_loss / args.num_steps
            accuracy = 100 * correct / total  # 修正：accuracy 单位为百分比
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

def validate(model, val_loader, criterion, device, epoch, args):
    """验证模型"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    json_logger.log(
        step=(epoch,),
        data={
            "val_loss": avg_loss,
            "accuracy": accuracy
        },
        verbosity=Verbosity.DEFAULT
    )
    return accuracy

def main():
    """主函数，执行训练流程"""
    args = get_args()
    device = setup_device()
    train_loader, val_loader = get_data_loaders(args)
    model, criterion, optimizer, scheduler = setup_model_and_optimizer(device, args)
    os.makedirs(args.save_path, exist_ok=True)
    best_acc = 0.0
    for epoch in range(args.epochs):
        json_logger.log(
            step=(epoch,),
            data={"epoch_start": f"Starting Epoch {epoch+1}"},
            verbosity=Verbosity.DEFAULT
        )
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args)
        acc = validate(model, val_loader, criterion, device, epoch, args)
        
        scheduler.step()
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.save_path, 'densenet_best.pth'))
            json_logger.log(
                step=(epoch,),
                data={"model_save": f"Saved best model, Accuracy: {best_acc:.2f}%"},
                verbosity=Verbosity.DEFAULT
            )

    torch.save(model.state_dict(), os.path.join(args.save_path, 'densenet_final.pth'))
    json_logger.log(
        step=(),
        data={"training_completed": "Training completed!"},
        verbosity=Verbosity.DEFAULT
    )

if __name__ == '__main__':
    main()