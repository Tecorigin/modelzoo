# Adapted to tecorigin hardware。

import os
import sys
from tqdm import tqdm
import time
import json

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder


train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_datasets(dataset_path):
    train_dataset = ImageFolder(os.path.join(dataset_path, 'train'), train_transforms)
    val_dataset = ImageFolder(os.path.join(dataset_path, 'val'), val_transforms)
    
    return train_dataset, val_dataset

def collate_fn(batch):
    images, labels = tuple(zip(*batch))

    images = torch.stack(images, dim=0)
    labels = torch.as_tensor(labels)
    return images, labels


def calculate_accuracy(outputs, targets):
    _, max5 = torch.topk(outputs, 5, dim=-1)
    total = targets.size(0)
    targets = targets.view(-1, 1)
    
    top1 = (targets == max5[:, 0:1]).sum().item()
    top5 = (targets == max5).sum().item()
    
    return top1, top5, total


def train_one_epoch(model, optimizer, scaler, data_loader, device, epoch, use_acm, rank, local_rank, img_size, lr, train_throughput, max_step, save_path):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    
    total_data_to_device_time = 0
    total_forward_time = 0
    total_backward_time = 0
    total_optimizer_step_time = 0
    start_time = time.time()
    global_step = 0
    for step, data in enumerate(data_loader):
        batch_start_time = time.time()
        if max_step > 0 and global_step >= max_step:
            break
        images, labels = data
        # images, labels = images.to(device).to(memory_format=torch.channels_last), labels.to(device)
        images, labels = images.to(device), labels.to(device)
        
        data_to_device_time = time.time() - batch_start_time
        total_data_to_device_time += data_to_device_time
        
        # 记录前向计算时间
        forward_start_time = time.time()
        if use_acm:
            with torch.sdaa.amp.autocast():
                sample_num += images.shape[0]

                pred = model(images)
                pred_classes = torch.max(pred, dim=1)[1]
                tmp_accu = torch.eq(pred_classes, labels).sum()
                accu_num += tmp_accu

                loss = loss_function(pred, labels)
        else:
            sample_num += images.shape[0]

            pred = model(images)
            pred_classes = torch.max(pred, dim=1)[1]
            tmp_accu = torch.eq(pred_classes, labels).sum()
            accu_num += tmp_accu

            loss = loss_function(pred, labels)
        total_forward_time = time.time() - forward_start_time
        
        optimizer.zero_grad()
        
        # 记录反向传播时间
        backward_start_time = time.time()
        scaler.scale(loss).backward()    # loss缩放并反向传播
        total_backward_time = time.time() - backward_start_time
        
        # 记录优化器步骤时间
        optimizer_step_start_time = time.time()
        scaler.step(optimizer)    # 参数更新
        scaler.update()    # 基于动态Loss Scale更新loss_scaling系数
        total_optimizer_step_time = time.time() - optimizer_step_start_time
        if local_rank == 0 or local_rank == -1:
            with open(os.path.join(save_path, 'log.jsonl'), 'a') as f:
                json.dump({'Iteration': step, 'Loss': float(loss.detach()), 'Acc': float(tmp_accu / images.shape[0])}, f)
                f.write('\n')
            if 'scripts' in os.getcwd():
                with open(os.path.join(os.getcwd(), '../log.jsonl'), 'a') as f:
                    json.dump({'Iteration': step, 'Loss': float(loss.mean().detach()), 'Acc': float(tmp_accu / images.shape[0])}, f)
                    f.write('\n')
            else:
                with open(os.path.join(os.getcwd(), 'log.jsonl'), 'a') as f:
                    json.dump({'Iteration': step, 'Loss': float(loss.mean().detach()), 'Acc': float(tmp_accu / images.shape[0])}, f)
                    f.write('\n')
            
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                            accu_loss.item() / (step + 1),
                                                                            accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        train_time = time.time() - start_time
        if train_time > 2 * 60 * 60:
            sys.exit(1)
        train_throughput = train_throughput / train_time
        global_step += 1
    total_time = time.time() - start_time
    compute_time = total_time - total_data_to_device_time
    
    return accu_loss.item() / len(data_loader), accu_num.item() / sample_num, total_data_to_device_time, compute_time, total_forward_time, total_backward_time, total_optimizer_step_time

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num