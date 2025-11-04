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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import evaluate
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

def fit_one_epoch(model_train, model, loss_history, loss, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, sdaa, test_loader, Batch_size, lfw_eval_flag, fp16, scaler, save_period, save_dir, local_rank):
    total_triple_loss   = 0
    total_CE_loss       = 0
    total_accuracy      = 0

    val_total_triple_loss   = 0
    val_total_CE_loss       = 0
    val_total_accuracy      = 0
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
        images, labels = batch
        data_end_time = time.time()
        data_times.append(data_end_time - data_start_time)

        with torch.no_grad():
            if sdaa:
                local_rank = int(os.environ.get("LOCAL_RANK", -1)) 
                device = torch.device(f"sdaa:{local_rank}")
                images  = images.to(device).to(memory_format=torch.channels_last)
                labels  = labels.to(device)

        optimizer.zero_grad()

        fp_start_time = time.time()
        if not fp16:
            outputs1, outputs2 = model_train(images, "train")
            fp_end_time = time.time()
            fp_times.append(fp_end_time - fp_start_time)
            bp_start_time = fp_end_time

            _triplet_loss   = loss(outputs1, Batch_size)
            _CE_loss        = nn.NLLLoss()(F.log_softmax(outputs2, dim = -1), labels)
            _loss           = _triplet_loss + _CE_loss #facenet

            _loss.backward()
            bp_end_time = time.time()  
            bp_times.append(bp_end_time - bp_start_time) 
            grad_start_time = bp_start_time
            optimizer.step()
            grad_end_time = time.time() 
            grad_times.append(grad_end_time - grad_start_time)
        else:
            # from torch.sdaa.amp import autocast
            with torch_sdaa.amp.autocast():
                outputs1, outputs2 = model_train(images, "train")
                fp_end_time = time.time()
                fp_times.append(fp_end_time - fp_start_time)
                bp_start_time = fp_end_time

                _triplet_loss   = loss(outputs1, Batch_size)
                _CE_loss        = nn.NLLLoss()(F.log_softmax(outputs2, dim = -1), labels)
                _loss           = _triplet_loss + _CE_loss
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(_loss).backward()
            bp_end_time = time.time()  
            bp_times.append(bp_end_time - bp_start_time) 
            grad_start_time = bp_start_time
            scaler.step(optimizer)
            scaler.update()
            grad_end_time = time.time() 
            grad_times.append(grad_end_time - grad_start_time)  
        
        

        with torch.no_grad():
            accuracy         = torch.mean((torch.argmax(F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor)) #facenet
    
            
        total_triple_loss   += _triplet_loss.item()
        total_CE_loss       += _CE_loss.item()
        total_accuracy      += accuracy.item() #facenet

        json_logger.log(
                    step = (epoch, iteration),
                    data = {
                            "total_triple_loss":total_triple_loss / (iteration + 1), 
                            "total_CE_loss":total_CE_loss / (iteration + 1), 
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
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, labels = batch
        val_samples += images.size(0)
        with torch.no_grad():
            if sdaa:
                local_rank = int(os.environ.get("LOCAL_RANK", -1)) 
                device = torch.device(f"sdaa:{local_rank}")
                images  = images.to(device).to(memory_format=torch.channels_last)
                labels  = labels.to(device)

            optimizer.zero_grad()
            outputs1, outputs2 = model_train(images, "train")
            
            _triplet_loss   = loss(outputs1, Batch_size)
            _CE_loss        = nn.NLLLoss()(F.log_softmax(outputs2, dim = -1), labels)
            _loss           = _triplet_loss + _CE_loss
            
            accuracy        = torch.mean((torch.argmax(F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))
            
            val_total_triple_loss   += _triplet_loss.item()
            val_total_CE_loss       += _CE_loss.item()
            val_total_accuracy      += accuracy.item()
        end_time_val = time.time()
        val_ips = val_samples / (end_time_val - start_time_val)
        val_time = end_time_val - start_time_val
        json_logger.log(
                    step = (epoch, iteration),
                    data = {
                            "val_total_triple_loss":val_total_triple_loss / (iteration + 1),
                            "val_total_CE_loss":val_total_CE_loss / (iteration + 1),
                            "val_accuracy":val_total_accuracy / (iteration + 1), 
                            "val_ips":val_ips,
                            "val_time": val_time,
                            },
                    verbosity=Verbosity.DEFAULT,
                )


        
    if lfw_eval_flag:
        print("开始进行LFW数据集的验证。")
        labels, distances = [], []
        for _, (data_a, data_p, label) in enumerate(test_loader):
            with torch.no_grad():
                data_a, data_p = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
                if sdaa:
                    local_rank = int(os.environ.get("LOCAL_RANK", -1)) 
                    device = torch.device(f"sdaa:{local_rank}")
                    data_a, data_p = data_a.to(device), data_p.to(device)
                    # data_a, data_p = data_a.sdaa(local_rank), data_p.sdaa(local_rank)
                out_a, out_p = model_train(data_a), model_train(data_p)
                dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
            distances.append(dists.data.cpu().numpy())
            labels.append(label.data.cpu().numpy())
        
        labels      = np.array([sublabel for label in labels for sublabel in label])
        distances   = np.array([subdist for dist in distances for subdist in dist])
        _, _, accuracy, _, _, _, _ = evaluate(distances,labels)
        
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')

        if lfw_eval_flag:
            print('LFW_Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))

        loss_history.append_loss(epoch, np.mean(accuracy) if lfw_eval_flag else total_accuracy / epoch_step, \
            (total_triple_loss + total_CE_loss) / epoch_step, (val_total_triple_loss + val_total_CE_loss) / epoch_step_val)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.4f' % ((total_triple_loss + total_CE_loss) / epoch_step))
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1),
                                                                    (total_triple_loss + total_CE_loss) / epoch_step,
                                                                    (val_total_triple_loss + val_total_CE_loss) / epoch_step_val))) 
