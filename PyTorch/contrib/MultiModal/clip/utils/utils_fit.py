import math
import os
from copy import deepcopy
import torch_sdaa
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import time
import torch.distributed as dist
from .callbacks import de_parallel
from .utils import get_lr
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity


def fit_one_epoch(model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, json_logger, distributed, local_rank=0):
    total_loss      = 0
    val_total_loss  = 0
    device = torch.device(f"sdaa:{local_rank}")
    if local_rank == 0:
        print('Start Train')
    model_train = model.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        start_time = time.time()
        images, texts = batch

        with torch.no_grad():
            if cuda:
                images = images.to(device)
                print(images[0].shape)
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            fp_time = time.time()
            # 这里不使用logits_per_text是因为dp模式的划分有问题，所以使用logits_per_image出来的后转置。
            logits_per_image, _                 = model_train(images, texts)
            logits_per_text                     = logits_per_image.t()
            labels                              = torch.arange(len(logits_per_image)).long().to(images.device)

            loss_logits_per_image               = nn.CrossEntropyLoss()(logits_per_image, labels)
            loss_logits_per_text                = nn.CrossEntropyLoss()(logits_per_text, labels)
            loss                                = loss_logits_per_image + loss_logits_per_text
            fp_timeend = time.time() - fp_time

            bp_time = time.time()
            loss.backward()
            bp_timeend = time.time()-bp_time

            grad_time = time.time()
            optimizer.step()
            grad_timeend = time.time() - grad_time

        else:
            model_train = model.train()
            # from torch_sdaa.amp import autocast
            with torch_sdaa.amp.autocast():
                fp_time = time.time()
                logits_per_image, _     = model_train(images, texts)
                logits_per_text         = logits_per_image.t()
                labels                              = torch.arange(len(logits_per_image)).long().to(images.device)
                logits_per_image = logits_per_image.float()
                logits_per_text = logits_per_text.float()

                loss_logits_per_image               = nn.CrossEntropyLoss()(logits_per_image, labels)
                loss_logits_per_text                = nn.CrossEntropyLoss()(logits_per_text, labels)
                loss                                = loss_logits_per_image + loss_logits_per_text
                fp_timeend = time.time() - fp_time
            #----------------------#
            #   反向传播
            #----------------------#
            bp_time = time.time()
            scaler.scale(loss).backward()
            bp_timeend = time.time() - bp_time

            grad_time = time.time()
            scaler.step(optimizer)
            scaler.update()
            grad_timeend = time.time() - grad_time
            
        total_loss += loss.item()

        with torch.no_grad():
            de_parallel(model_train).logit_scale.clamp_(0, math.log(100))

        iteration_time = time.time() - start_time

        if distributed:
            json_logger.log(
                step=(epoch, iteration),
                data={
                    "rank": dist.get_rank(),
                    "train.loss": 5 * total_loss / (iteration + 1),
                    "train.ips": int(len(images) / iteration_time),
                    "train.lr": get_lr(optimizer),
                    "data.shape": images[0].shape,
                    "train.compute_time": iteration_time,
                    "train.fp_time": fp_timeend,
                    "train.bp_time": bp_timeend,
                    "train.grad_time": grad_timeend
                },
                verbosity=Verbosity.DEFAULT,
            )
        else:
            json_logger.log(
                step=(epoch, iteration),
                data={
                    "train.loss": 5 * total_loss / (iteration + 1),
                    "train.ips": int(len(images) / iteration_time),
                    "train.lr": get_lr(optimizer),
                    "data.shape": images[0].shape,
                    "train.compute_time": iteration_time,
                    "train.fp_time": fp_timeend,
                    "train.bp_time": bp_timeend,
                    "train.grad_time": grad_timeend
                },
                verbosity=Verbosity.DEFAULT,
            )

    if local_rank == 0:
        print('Finish Train')
        print('Start Validation')

    model_eval = model.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        start_time = time.time()
        images, texts = batch
        with torch.no_grad():
            if cuda:
                images = images.to(device)
            logits_per_image, _                 = model_eval(images, texts)
            logits_per_text                     = logits_per_image.t()
            labels                              = torch.arange(len(logits_per_image)).long().to(images.device)
            loss_logits_per_image               = nn.CrossEntropyLoss()(logits_per_image, labels)
            loss_logits_per_text                = nn.CrossEntropyLoss()(logits_per_text, labels)
            loss                                = loss_logits_per_image + loss_logits_per_text
            
            val_total_loss += loss.item()

        if local_rank == 0:
            iteration_time = time.time() - start_time
            json_logger.log(
                step=(epoch, iteration),
                data={
                    "val.loss": 5 * val_total_loss / (iteration + 1),
                    "val.ips": int(len(images)  / iteration_time),
                    "val.lr": get_lr(optimizer)
                },
                verbosity=Verbosity.DEFAULT,
            )


    if local_rank == 0:
        print('Finish Validation')

        loss_history.append_loss(epoch, 5* total_loss / epoch_step, 5* val_total_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_eval, epoch_step,json_logger)

        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_total_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(deepcopy(model).half().state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_total_loss / epoch_step_val)))
            
        if len(loss_history.val_loss) <= 1 or (val_total_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(deepcopy(model).half().state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(deepcopy(model).half().state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
