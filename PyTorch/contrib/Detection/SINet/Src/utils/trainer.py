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
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

import torch
from torch.autograd import Variable
from torch.cuda.amp import autocast
from datetime import datetime
import os
# from apex import amp
import torch.nn.functional as F
import time
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity

# 初始化 logger
json_logger = Logger([
    StdOutBackend(Verbosity.DEFAULT),
    JSONStreamBackend(Verbosity.VERBOSE, 'train_log.json'),
])
json_logger.metadata("train.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.ips", {"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})

def eval_mae(y_pred, y):
    """
    evaluate MAE (for test or validation phase)
    :param y_pred:
    :param y:
    :return: Mean Absolute Error
    """
    return torch.abs(y_pred - y).mean()


def numpy2tensor(numpy):
    """
    convert numpy_array in cpu to tensor in gpu
    :param numpy:
    :return: torch.from_numpy(numpy).cuda()
    """
    return torch.from_numpy(numpy).cuda()


def clip_gradient(optimizer, grad_clip):
    """
    recalibrate the misdirection in the training
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


def trainer(train_loader, model, optimizer, epoch, opt, loss_func, total_step, scaler=None):
    """
    Training iteration
    :param train_loader:
    :param model:
    :param optimizer:
    :param epoch:
    :param opt:
    :param loss_func:
    :param total_step:
    :param scaler:
    :return:
    """
    model.train()
    
    for step, data_pack in enumerate(train_loader):
        if step >= 100:  # 提前终止（调试用）
            print(f"[Info] Step limit reached: stopping at step {step}")
            break

        step_start_time = time.time()

        optimizer.zero_grad()
        images, gts = data_pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()

        with autocast():
            cam_sm, cam_im = model(images)
            loss_sm = loss_func(cam_sm, gts)
            loss_im = loss_func(cam_im, gts)
            loss_total = loss_sm + loss_im

        # with amp.scale_loss(loss_total, optimizer) as scale_loss:
        #     scale_loss.backward()
        scaler.scale(loss_total).backward()
        scaler.unscale_(optimizer)  # 可选：用于 gradient clipping 之前
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
        scaler.step(optimizer)
        scaler.update()
        # clip_gradient(optimizer, opt.clip)
        # optimizer.step()

        # 计算 ips (images per second)
        elapsed_time = time.time() - step_start_time
        ips = opt.batchsize / elapsed_time

        # 日志输出
        json_logger.log(
            step=(epoch, step),
            data={
                "rank": int(os.environ.get("LOCAL_RANK", 0)),  # 非DDP情况下为0
                "train.loss": loss_total.item(),
                "train.ips": ips
            },
            verbosity=Verbosity.DEFAULT
        )

        if step % 10 == 0 or step == total_step:
            print('[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => [Loss_s: {:.4f} Loss_i: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, step, total_step, loss_sm.data, loss_im.data))

    save_path = opt.save_model
    os.makedirs(save_path, exist_ok=True)

    if (epoch+1) % opt.save_epoch == 0:
        torch.save(model.state_dict(), save_path + 'SINet_%d.pth' % (epoch+1))
