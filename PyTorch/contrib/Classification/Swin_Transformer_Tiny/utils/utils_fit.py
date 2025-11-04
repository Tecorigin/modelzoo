import os
from threading import local
import torch_sdaa
import time

import torch
import torch.nn.functional as F
from torch import nn
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


def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, sdaa, fp16, scaler, save_period, save_dir, Batch_size, local_rank=0):
    total_loss      = 0
    total_accuracy  = 0

    val_loss        = 0
    val_accuracy    = 0
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
        images, targets = batch
        data_end_time = time.time()
        data_times.append(data_end_time - data_start_time)
        with torch.no_grad():
            if sdaa:
                local_rank = int(os.environ.get("LOCAL_RANK", -1)) 
                device = torch.device(f"sdaa:{local_rank}")
                images  = images.to(device).to(memory_format=torch.channels_last)
                targets  = targets.to(device)
                
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        fp_start_time = time.time()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs     = model_train(images)
            fp_end_time = time.time()
            fp_times.append(fp_end_time - fp_start_time)
            bp_start_time = fp_end_time
            #----------------------#
            #   计算损失
            #----------------------#
            loss_value  = nn.CrossEntropyLoss()(outputs, targets)
            loss_value.backward()
            bp_end_time = time.time()  
            bp_times.append(bp_end_time - bp_start_time) 
            grad_start_time = bp_start_time
            optimizer.step()
            grad_end_time = time.time() 
            grad_times.append(grad_end_time - grad_start_time)
        else:
            # from torch.sdaa.amp import autocast
            with torch_sdaa.amp.autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs     = model_train(images)
                fp_end_time = time.time()
                fp_times.append(fp_end_time - fp_start_time)
                bp_start_time = fp_end_time
                #----------------------#
                #   计算损失
                #----------------------#
                loss_value  = nn.CrossEntropyLoss()(outputs, targets)
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss_value).backward()
            bp_end_time = time.time()  
            bp_times.append(bp_end_time - bp_start_time) 
            grad_start_time = bp_start_time
            scaler.step(optimizer)
            scaler.update()
            grad_end_time = time.time() 
            grad_times.append(grad_end_time - grad_start_time)  

        total_loss += loss_value.item()
        with torch.no_grad():
            accuracy = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
            total_accuracy += accuracy.item()

        # if local_rank == 0:
        #     pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
        #                         'accuracy'  : total_accuracy / (iteration + 1), 
        #                         'lr'        : get_lr(optimizer)})
        #     pbar.update(1)
        json_logger.log(
                    step = (epoch, iteration),
                    data = {
                            'total_loss': total_loss / (iteration + 1),
                            'accuracy'  : total_accuracy / (iteration + 1), 
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
        images, targets = batch
        val_samples += images.size(0)
        with torch.no_grad():
            if sdaa:
                local_rank = int(os.environ.get("LOCAL_RANK", -1)) 
                device = torch.device(f"sdaa:{local_rank}")
                images  = images.to(device).to(memory_format=torch.channels_last)
                targets  = targets.to(device)

            optimizer.zero_grad()

            outputs     = model_train(images)
            loss_value  = nn.CrossEntropyLoss()(outputs, targets)
            
            val_loss    += loss_value.item()
            accuracy        = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
            val_accuracy    += accuracy.item()
        end_time_val = time.time()
        val_ips = val_samples / (end_time_val - start_time_val)
        val_time = end_time_val - start_time_val

            
        # if local_rank == 0:
        #     pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
        #                         'accuracy'  : val_accuracy / (iteration + 1), 
        #                         'lr'        : get_lr(optimizer)})
        #     pbar.update(1)
        json_logger.log(
                    step = (epoch, iteration),
                    data = {
                            'val_total_loss': val_loss / (iteration + 1),
                            "val_accuracy": val_accuracy / (iteration + 1), 
                            "val_ips":val_ips,
                            "val_time": val_time,
                            },
                    verbosity=Verbosity.DEFAULT,
                )

                
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
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
