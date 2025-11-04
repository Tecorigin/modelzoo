import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import argparse
import time
import datetime
from tcap_dllogger import Logger, StdOutBackend, Verbosity
from torch.sdaa import amp
from gmlp import VisiongMLP
from configs import configs

scaler = torch.sdaa.amp.GradScaler()

json_logger = Logger(
    [
        StdOutBackend(Verbosity.DEFAULT),
    ]
)

json_logger.metadata("train.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.ips", {"unit": "imgs/s", "format": "%.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("val_loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "VAL"})
json_logger.metadata("accuracy", {"unit": "%", "GOAL": "MAXIMIZE", "STAGE": "VAL"})
json_logger.metadata("device", {"unit": "INFO", "GOAL": "NONE", "STAGE": "INFO"})
json_logger.metadata("epoch_start", {"unit": "INFO", "GOAL": "NONE", "STAGE": "TRAIN"})
json_logger.metadata("model_save", {"unit": "INFO", "GOAL": "NONE", "STAGE": "TRAIN"})
json_logger.metadata("training_completed", {"unit": "INFO", "GOAL": "NONE", "STAGE": "TRAIN"})

def setup_model_and_optimizer(device, args):
    """初始化模型、损失函数、优化器和调度器"""
    model = VisiongMLP(**configs["Ti"]).to(device)  # 移除重复的 prob_0_L 参数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    return model, criterion, optimizer, scheduler

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train VisiongMLP on CIFAR-10 with SDAA')
    parser.add_argument('--data_path', type=str, default='/data/teco-data/cifar10', help='Path to CIFAR-10 dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Initial learning rate')
    parser.add_argument('--save_path', type=str, default='./checkpoints', help='Model save path')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of training steps')
    return parser.parse_args()

def setup_device():
    """设置设备（SDAA或CPU）"""
    device = torch.device('sdaa' if torch.sdaa.is_available() else 'cpu')
    json_logger.log(
        step=(),
        data={"device": f"Using device: {device}"},
        verbosity=Verbosity.DEFAULT
    )
    return device

def get_data_loaders(args):
    """加载和预处理 CIFAR-10 数据集"""
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    return train_loader, val_loader

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
        with torch.sdaa.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
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
    '''
        acc = validate(model, val_loader, criterion, device, epoch, args)
        
        scheduler.step()
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.save_path, 'visiongmlp_best.pth'))
            json_logger.log(
                step=(epoch,),
                data={"model_save": f"Saved best model, Accuracy: {best_acc:.2f}%"},
                verbosity=Verbosity.DEFAULT
            )

    torch.save(model.state_dict(), os.path.join(args.save_path, 'visiongmlp_final.pth'))
    json_logger.log(
        step=(),
        data={"training_completed": "Training completed!"},
        verbosity=Verbosity.DEFAULT
    )
    '''

if __name__ == '__main__':
    main()