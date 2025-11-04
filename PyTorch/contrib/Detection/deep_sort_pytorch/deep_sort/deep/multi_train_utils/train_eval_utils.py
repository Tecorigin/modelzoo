import sys
# Adapted to tecorigin hardware
import os
#导入sdaa库
import torch_sdaa
from tqdm import tqdm
import torch

from .distributed_utils import reduce_value, is_main_process

from torch.sdaa import amp              # 导入AMP
scaler = torch.sdaa.amp.GradScaler() 

def load_model(state_dict, model_state_dict, model):
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape {}, ' \
                      'loaded shape {}.'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    sum_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for idx, (images, labels) in enumerate(data_loader):
        # forward
        images, labels = images.to(device), labels.to(device)
        #使用混合精度
        images= images.to(memory_format=torch.channels_last) # 数据格式转换为NHWC（channels_last）
        
        # with torch.sdaa.amp.autocast():   # 开启AMP环境
        #     outputs = model(imgs)    
        #     loss = loss_func(outputs, labels) 
        # optimizer.zero_grad()
        # scaler.scale(loss).backward()    # loss缩放并反向转播
        # scaler.step(optimizer)    # 参数更新
        # scaler.update()    # 基于动态Loss Scale更新loss_scaling系数
        with torch.sdaa.amp.autocast():
            outputs = model(images)# 开启AMP环境
            loss = criterion(outputs, labels)
        optimizer.zero_grad()
        scaler.scale(loss).backward()    # loss缩放并反向转播
        
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * idx + loss.detach()) / (idx + 1)
        pred = torch.max(outputs, dim=1)[1]
        sum_num += torch.eq(pred, labels).sum()

        if is_main_process():
            data_loader.desc = '[epoch {}] mean loss {}'.format(epoch, mean_loss.item())

        if not torch.isfinite(loss):
            print('loss is infinite, ending training')
            sys.exit(1)

        scaler.step(optimizer)    # 参数更新
        scaler.update()    # 基于动态Loss Scale更新loss_scaling系数
            
        # outputs = model(images)
        # loss = criterion(outputs, labels)
        # backward
        # loss.backward()
        # loss = reduce_value(loss, average=True)
        # mean_loss = (mean_loss * idx + loss.detach()) / (idx + 1)
        # pred = torch.max(outputs, dim=1)[1]
        # sum_num += torch.eq(pred, labels).sum()

        # if is_main_process():
        #     data_loader.desc = '[epoch {}] mean loss {}'.format(epoch, mean_loss.item())

        # if not torch.isfinite(loss):
        #     print('loss is infinite, ending training')
        #     sys.exit(1)

        # optimizer.step()
        # optimizer.zero_grad()
    if device != torch.device('cpu'):
        #改用sdaa
        # torch.cuda.synchronize(device)
        torch.sdaa.synchronize(device)
    sum_num = reduce_value(sum_num, average=False)

    return sum_num.item(), mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = torch.zeros(1).to(device)
    sum_num = torch.zeros(1).to(device)
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for idx, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        #使用混合精度
        inputs= inputs.to(memory_format=torch.channels_last) # 数据格式转换为NHWC（channels_last）
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss = reduce_value(loss, average=True)

        test_loss = (test_loss * idx + loss.detach()) / (idx + 1)
        pred = torch.max(outputs, dim=1)[1]
        sum_num += torch.eq(pred, labels).sum()

    if device != torch.device('cpu'):
        #改用sdaa
        # torch.cuda.synchronize(device)
        torch.sdaa.synchronize(device)
        

    sum_num = reduce_value(sum_num, average=False)

    return sum_num.item(), test_loss.item()
