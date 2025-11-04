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
import os
import time
import torch
from tqdm import tqdm

from utils.utils import get_lr
# 初始化logger
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity

json_logger = Logger(
    [
        StdOutBackend(Verbosity.DEFAULT),
        JSONStreamBackend(Verbosity.VERBOSE, "./logs/dlloger_example.json"),
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


def fit_one_epoch(device, model, train_util, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, use_sdaa, use_amp, scaler, save_period, save_dir, batch_size, local_rank):
    total_loss = 0
    rpn_loc_loss = 0
    rpn_cls_loss = 0
    roi_loc_loss = 0
    roi_cls_loss = 0
    val_loss = 0
    start_time = time.time()
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    # with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, boxes, labels = batch[0], batch[1], batch[2]
        with torch.no_grad():
            if use_sdaa:
                images = images.to(device)

        rpn_loc, rpn_cls, roi_loc, roi_cls, total = train_util.train_step(images, boxes, labels, 1, use_amp, scaler)
        total_loss += total.item()
        rpn_loc_loss += rpn_loc.item()
        rpn_cls_loss += rpn_cls.item()
        roi_loc_loss += roi_loc.item()
        roi_cls_loss += roi_cls.item()

        if local_rank == 0:
            pbar.set_postfix(**{'train.loss': total_loss / (iteration + 1),
                                'rpn_loc': rpn_loc_loss / (iteration + 1),
                                'rpn_cls': rpn_cls_loss / (iteration + 1),
                                'roi_loc': roi_loc_loss / (iteration + 1),
                                'roi_cls': roi_cls_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
            json_logger.log(
                step=(epoch, iteration),
                data={
                    'train.loss': total_loss / (iteration + 1),
                    'rpn_loc': rpn_loc_loss / (iteration + 1),
                    'rpn_cls': rpn_cls_loss / (iteration + 1),
                    'roi_loc': roi_loc_loss / (iteration + 1),
                    'roi_cls': roi_cls_loss / (iteration + 1),
                    "train.lr": get_lr(optimizer),
                    'train.ips': (iteration + 1) * batch_size / (time.time() - start_time)
                },
                verbosity=Verbosity.DEFAULT,
            )

    if local_rank == 0:
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d.pth' % (
            epoch + 1)))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
