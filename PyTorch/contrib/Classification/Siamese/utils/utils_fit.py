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
import torch_sdaa

import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import get_lr
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity

json_logger = Logger(
[
    StdOutBackend(Verbosity.DEFAULT),
    JSONStreamBackend(Verbosity.VERBOSE, 'dlloger_example.json'),
]
)

json_logger.metadata("train.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.loss_mean", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("val.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "VALID"})
json_logger.metadata("train.ips",{"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("val.ips",{"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "VALID"})
json_logger.metadata("train.compute_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.fp_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.bp_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.grad_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})


def fit_one_epoch(model_train, model, loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, genval, Epoch, sdaa, fp16, scaler, save_period, save_dir, Batch_size, local_rank=0):
    total_loss      = 0
    total_accuracy  = 0

    val_loss            = 0
    val_total_accuracy  = 0
    start_time = time.time()

    
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()

    # 记录训练时间
    data_times = []
    compute_times = []
    fp_times = []
    bp_times = []
    grad_times = []

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        data_start_time = time.time()
        images, targets = batch[0], batch[1]
        data_end_time = time.time()
        data_times.append(data_end_time - data_start_time)
        with torch.no_grad():
            if sdaa:
                local_rank = int(os.environ.get("LOCAL_RANK", -1)) 
                device = torch.device(f"sdaa:{local_rank}")
                images  = images.to(device)
                targets = targets.to(device)

        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        fp_start_time = time.time()
        if not fp16:
            outputs = model_train(images)
            fp_end_time = time.time()
            fp_times.append(fp_end_time - fp_start_time)
            bp_start_time = fp_end_time

            output  = loss(outputs, targets)

            output.backward()
            bp_end_time = time.time()  
            bp_times.append(bp_end_time - bp_start_time) 
            grad_start_time = bp_start_time
            optimizer.step()
            grad_end_time = time.time() 
            grad_times.append(grad_end_time - grad_start_time)
        else:
            with torch_sdaa.amp.autocast():
                outputs = model_train(images)
                fp_end_time = time.time()
                fp_times.append(fp_end_time - fp_start_time)
                bp_start_time = fp_end_time

                output  = loss(outputs, targets)
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(output).backward()
            bp_end_time = time.time()  
            bp_times.append(bp_end_time - bp_start_time) 
            grad_start_time = bp_start_time
            scaler.step(optimizer)
            scaler.update()
            grad_end_time = time.time() 
            grad_times.append(grad_end_time - grad_start_time)  

        with torch.no_grad():
            equal       = torch.eq(torch.round(nn.Sigmoid()(outputs).float()), targets)
            accuracy    = torch.mean(equal.float())

        total_loss      += output.item()
        total_accuracy  += accuracy.item()

        # if local_rank == 0:
        #     pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
        #                         'acc'       : total_accuracy / (iteration + 1),
        #                         'lr'        : get_lr(optimizer)})
        #     pbar.update(1)
        
        json_logger.log(
                    step = (epoch, iteration),
                    data = {
                            "total_loss":total_loss / (iteration + 1), 
                            "accuracy":total_accuracy / (iteration + 1), 
                            "train.lr":get_lr(optimizer),
                            "train.data_time": sum(data_times) / len(data_times) if data_times else 0,
                            "train.compute_time": sum(fp_times + bp_times) / len(fp_times + bp_times) if fp_times + bp_times else 0,
                            "train.fp_time": sum(fp_times) / len(fp_times) if fp_times else 0,
                            "train.bp_time": sum(bp_times) / len(bp_times) if bp_times else 0,
                            "train.grad_time": sum(grad_times) / len(grad_times) if grad_times else 0,
                            "train.ips": (iteration + 1) * Batch_size / (time.time() - start_time)
                            },
                    verbosity=Verbosity.DEFAULT,
                )

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.eval()
    start_time_val = time.time()
    val_samples = 0
    for iteration, batch in enumerate(genval):
        if iteration >= epoch_step_val:
            break
        
        images, targets = batch[0], batch[1]
        val_samples = val_samples + 1
        with torch.no_grad():
            if sdaa:
                local_rank = int(os.environ.get("LOCAL_RANK", -1)) 
                device = torch.device(f"sdaa:{local_rank}")
                images  = images.to(device)
                targets = targets.to(device)
                
            optimizer.zero_grad()
            outputs = model_train(images)
            output  = loss(outputs, targets)

            equal       = torch.eq(torch.round(nn.Sigmoid()(outputs)), targets)
            accuracy    = torch.mean(equal.float())

        val_loss            += output.item()
        val_total_accuracy  += accuracy.item()
        end_time_val = time.time()
        val_ips = val_samples / (end_time_val - start_time_val)
        val_time = end_time_val - start_time_val

        # if local_rank == 0:
        #     pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1), 
        #                         'acc'       : val_total_accuracy / (iteration + 1)})
        #     pbar.update(1)
        
        json_logger.log(
                    step = (epoch, iteration),
                    data = {
                            "val_loss":val_loss / (iteration + 1),
                            "val_accuracy":val_total_accuracy / (iteration + 1), 
                            "val_ips":val_ips,
                            "val_time": val_time,
                            },
                    verbosity=Verbosity.DEFAULT,
                )
                
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))