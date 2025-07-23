import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
from config import load_config
from preprocess import load_data
from model import LambdaResNet18, get_n_params

import torch_sdaa
# from torch.sdaa import amp

import datetime
import time  # 添加时间模块以计算 IPS

from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity

json_logger = Logger(
    [
        StdOutBackend(Verbosity.DEFAULT),
        JSONStreamBackend(Verbosity.VERBOSE, "tmp.json"),
    ]
)

json_logger.metadata("train.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.ips", {"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})

device = torch.device("sdaa" if torch.sdaa.is_available() else "cpu")

def save_checkpoint(best_acc, model, optimizer, args, epoch):
    print('Best Model Saving...')
    if args.device_num > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    torch.save({
        'model_state_dict': model_state_dict,
        'global_epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }, os.path.join('checkpoints', 'checkpoint_model_best.pth'))


def _train(epoch, train_loader, model, optimizer, criterion, args):
    print("start training...")
    model.train()

    i = 0
    losses = 0.
    acc = 0.
    total = 0.
    # scaler = torch.sdaa.amp.GradScaler()
    
    for idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.to(device), target.to(device)

        output = model(data)
        _, pred = F.softmax(output, dim=-1).max(1)
        acc += pred.eq(target).sum().item()
        total += target.size(0)

        optimizer.zero_grad()

        # 记录开始时间
        start_time = time.time()
        
        loss = criterion(output, target)
        losses += loss
        
        loss.backward()
        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        
        # 计算 IPS（每秒处理的图片数）
        batch_time = time.time() - start_time
        ips = data.size(0) / batch_time if batch_time > 0 else 0
        

        # 打印信息
        # print('[Epoch: {0:4d}], [iteration: {1:4d}], Loss: {2:.3f}'.format(epoch, i, loss))
        
        # 使用 logger 输出指定格式
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
        
        if idx % args.print_intervals == 0 and idx != 0:
            print('[Epoch: {0:4d}], Loss: {1:.3f}, Acc: {2:.3f}, Correct {3} / Total {4}'.format(epoch,
                                                                                                 losses / (idx + 1),
                                                                                                 acc / total * 100.,
                                                                                                 acc, total))
        i += 1
        # print("i is :",i)
        # print("args.num_step:",args.num_steps)
        if(i== args.num_steps):
            break


def _eval(epoch, test_loader, model, args):
    model.eval()

    acc = 0.
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = F.softmax(output, dim=-1).max(1)

            acc += pred.eq(target).sum().item()
        print('[Epoch: {0:4d}], Acc: {1:.3f}'.format(epoch, acc / len(test_loader.dataset) * 100.))

    return acc / len(test_loader.dataset) * 100.


def main(args):
    train_loader, test_loader = load_data(args)
    model = LambdaResNet18()
    print('Model Parameters: {}'.format(get_n_params(model)))

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    if args.checkpoints is not None:
        checkpoints = torch.load(os.path.join('checkpoints', args.checkpoints))
        model.load_state_dict(checkpoints['model_state_dict'])
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
        start_epoch = checkpoints['global_epoch']
    else:
        start_epoch = 1

    if args.cuda:
        model = model.to(device)

    if not args.evaluation:
        criterion = nn.CrossEntropyLoss()
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.0001)

        global_acc = 0.
        for epoch in range(start_epoch, args.epochs + 1):
            _train(epoch, train_loader, model, optimizer, criterion, args)
            best_acc = _eval(epoch, test_loader, model, args)
            if global_acc < best_acc:
                global_acc = best_acc
                save_checkpoint(best_acc, model, optimizer, args, epoch)

            # lr_scheduler.step()
            # print('Current Learning Rate: {}'.format(lr_scheduler.get_last_lr()))
    else:
        _eval(start_epoch, test_loader, model, args)


if __name__ == '__main__':
    args = load_config()
    main(args)