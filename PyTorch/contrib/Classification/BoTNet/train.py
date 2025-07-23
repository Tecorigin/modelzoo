import logging
import warnings
import argparse
import datetime
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import csv
import os
import time
from torch.cuda.amp import GradScaler, autocast
from BoTNet_Model import resnet50

import torch_sdaa
from torch.sdaa import amp

from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity

# 配置 logger
json_logger = Logger(
    [
        StdOutBackend(Verbosity.DEFAULT),
        JSONStreamBackend(Verbosity.VERBOSE, 'dlloger_example.json'),
    ]
)

json_logger.metadata("train.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.ips", {"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})

CLASSES = 10

def prepare_data(img_size, batch_size):
    path = "/data/bupt-data/lzw/imagenette2-320/"
    
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    valid_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ImageFolder(root=path + 'train', transform=train_transforms)
    valid_dataset = ImageFolder(root=path + 'val', transform=valid_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, valid_loader

def run_training(train_loader, valid_loader, lr, epochs, output, num_heads, img_size, name, time_str, num_steps):
    print("Setup model..")
    model = resnet50(num_classes=CLASSES, attention=[False, False, False, True], num_heads=num_heads, image_size=img_size)
    
    device = torch.device("sdaa" if torch.sdaa.is_available() else "cpu")
    model = model.to(device)
    
    print("built model")
    print("Saving files to", output)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-8, threshold=0.001)
    
    csv_file = os.path.join(output, "BoTNet.csv") 
    os.makedirs(output, exist_ok=True)
    model_path = os.path.join(output, f'imagenette{time_str}_BotNet_{name}.pth')
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if os.path.getsize(csv_file) == 0:
            writer.writerow(['epoch', 'train_loss', 'valid_loss', 'accuracy'])
    i = 0
    step = 0
    best_acc = 0.0
    scaler = torch.sdaa.amp.GradScaler()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 1):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            start_time = time.time()
            
            with torch.sdaa.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            batch_time = time.time() - start_time
            ips = inputs.size(0) / batch_time if batch_time > 0 else 0
            
            train_loss += loss.item() * inputs.size(0)
            
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            log_message = (
                f"TCAPPDLL {current_time} - Epoch: {epoch+1} Iteration: {i}  "
                f"rank : 0  train.loss : {loss.item():.6f}  train.ips : {ips:.6f} imgs/s"
            )
            json_logger.log(
                step=(epoch, i),
                data={
                    "rank": 0,
                    "train.loss": loss.item(),
                    "train.ips": ips,
                },
                verbosity=Verbosity.DEFAULT,
            )
            # print("i",i)
            if(i==num_steps):
                return
            i += 1

        train_loss /= len(train_loader.dataset)
        
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        valid_loss /= len(valid_loader.dataset)
        accuracy = correct / total
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, valid_loss, accuracy])
        
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), model_path)
        
        scheduler.step(accuracy)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Accuracy: {accuracy:.4f}')

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--img_size', default=224, type=int,
                        help='Image size to which input images should be scaled.')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch Size for training.')
    parser.add_argument('--lr', default=0.0002, type=float,
                        help='Learning rate for training model.')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Epoch number for training model.')
    parser.add_argument('--output', default="./", type=str,
                        help='Where to output all stuff')
    parser.add_argument('--num_heads', default=4, type=int,
                        help='Number of heads in attention layers.')
    parser.add_argument('--name', default="", type=str,
                        help='Add a name ending to saved model and log file.')
    parser.add_argument('--num_steps', default=100, type=int,
                        help='train iteration limits')
    parser.add_argument('--time', default="", type=str,
                        help='Timestamp for model and log file naming.')
    args = parser.parse_args()
    
    print(args)
    time_str = args.time if args.time else str(datetime.datetime.now())[:-7].replace(" ", "_")
    print("Starting at " + time_str)

    train_loader, valid_loader = prepare_data(args.img_size, args.batch_size)
    print("Prepared Data")
    
    run_training(train_loader, valid_loader, args.lr, args.epochs, args.output, args.num_heads, args.img_size, args.name, time_str, args.num_steps)

if __name__ == '__main__':
    main()