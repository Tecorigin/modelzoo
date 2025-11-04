# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-      Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Adapted to tecorigin hardware

import time
from copy import deepcopy
from functools import wraps
from typing import Callable, Dict, Optional, Tuple
import os

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
from ..utils import logger as log
from ..utils import utils
from ..utils.logger import TrainingMetrics, ValidationMetrics
from ..models.common import EMA
import os
from ..utils.tools import get_tools, t_step, t_stop
try:
    import torch_sdaa
    import torch_sdaa.core.sdaa_model as sm
    import torch_sdaa.distributed as sdaa_dist
except:
    print('import torch_sdaa failed')

class Executor:
    def __init__(
        self,
        args,
        device,
        model: nn.Module,
        loss: Optional[nn.Module],
        memory_format: torch.memory_format = torch.contiguous_format,
        amp: bool = False,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        divide_loss: int = 1,
        ts_script: bool = False,
        tensorboard_list=None,
    ):
        # assert not (amp and scaler is None), "Gradient Scaler is needed for AMP"
        if amp and scaler is None:
            print("amp is used but Gradient Scaler enable")

        def xform(m: nn.Module) -> nn.Module:
            m = m.to(device)
            m.to(memory_format=memory_format)
            return m
        if args.model_cl:
            self.model = xform(model)
        else:
            self.model = model.to(device)
        if ts_script:
            self.model = torch.jit.script(self.model)
        self.ts_script = ts_script
        self.loss = xform(loss) if loss is not None else None
        if device.type == "cpu" and args.FP64:
            self.model = self.model.to(torch.float64)
        self.amp = amp
        self.scaler = scaler
        self.is_distributed = False
        self.divide_loss = divide_loss
        self._fwd_bwd = None
        self._forward = None
        self.args = args
        self.tensorboard_list = tensorboard_list
        self.step = 0
        self.device = device
        self.memory_format = memory_format

    def get_autocast(self,args):
        if args.device == 'sdaa':
            from torch_sdaa.amp import autocast
        else:
            from torch.cuda.amp import autocast
        return autocast
    
    def distributed(self, device, gpu_id):
        if device == 'sdaa':
            self.is_distributed = True
            print('id', sm.current_device())
            self.model = DDP(self.model, bucket_cap_mb=110)
        else:
            self.is_distributed = True
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self.model = DDP(self.model, device_ids=[
                                 gpu_id], output_device=gpu_id)
            torch.cuda.current_stream().wait_stream(s)

    def _fwd_bwd_fn(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        if self.args.channel_last:
            input = input.to(memory_format=self.memory_format)

        fp_start = time.time()
        autocast = self.get_autocast(self.args)
        with autocast(enabled=self.amp):
            output = self.model(input)
            loss = self.loss(output, target)
            loss /= self.divide_loss
        
        bp_start = time.time()

        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        step_time = time.time()
        
        return loss, (bp_start - fp_start), (step_time-bp_start), step_time

    def _forward_fn(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.args.channel_last:
            input = input.to(memory_format=self.memory_format)

        autocast = self.get_autocast(self.args)
        
        if input.device.type == "cpu":
            output = self.model(input)
            loss = None if self.loss is None else self.loss(output, target)
        else:
            with torch.no_grad(), autocast(enabled=self.amp):
                output = self.model(input)
                loss = None if self.loss is None else self.loss(output, target)

        return output if loss is None else loss, output

    def optimize(self, fn):
        return fn

    @property
    def forward_backward(self):
        if self._fwd_bwd is None:
            if self.loss is None:
                raise NotImplementedError(
                    "Loss must not be None for forward+backward step"
                )
            self._fwd_bwd = self.optimize(self._fwd_bwd_fn)
        return self._fwd_bwd

    @property
    def forward(self):
        if self._forward is None:
            self._forward = self.optimize(self._forward_fn)
        return self._forward

    def train(self):
        self.model.train()
        if self.loss is not None:
            self.loss.train()

    def eval(self):
        self.model.eval()
        if self.loss is not None:
            self.loss.eval()


class Trainer:
    def __init__(
        self,
        executor: Executor,
        optimizer: torch.optim.Optimizer,
        grad_acc_steps: int,
        ema: Optional[float] = None,
        writer=None,
        tensorboard_list=None,
    ):
        self.executor = executor
        self.optimizer = optimizer
        self.grad_acc_steps = grad_acc_steps
        self.use_ema = False
        if ema is not None:
            self.ema_executor = deepcopy(self.executor)
            self.ema = EMA(ema, self.ema_executor.model)
            self.use_ema = True

        try:
            self.optimizer.zero_grad(set_to_none=True)
        except:
            self.optimizer.zero_grad()
        
        self.steps_since_update = 0

        self.writer = writer
        self.tensorboard_list = tensorboard_list
        self.step = 0

    def train(self):
        self.executor.train()
        if self.use_ema:
            self.ema_executor.train()

    def eval(self):
        self.executor.eval()
        if self.use_ema:
            self.ema_executor.eval()

    def train_step(self, input, target, lr_schedule = None, step=None):

        loss, fp_time, bp_time, step_time = self.executor.forward_backward(
            input, target)

        if lr_schedule:
            lr_schedule.step()
        
        self.steps_since_update += 1
        if self.steps_since_update == self.grad_acc_steps:
            if self.executor.scaler is not None:
                self.executor.scaler.step(self.optimizer)
                self.executor.scaler.update()
            else:
                self.optimizer.step()
            self.steps_since_update = 0
            self.scale_value = self.executor.scaler.get_scale() if self.executor.amp  else 0 

        
        self.step += 1

        self.optimizer.zero_grad()

        iter_time = time.time()

        if self.use_ema:
            self.ema(self.executor.model, step=step)

        return loss, fp_time, bp_time, (iter_time - step_time)

    def validation_steps(self) -> Dict[str, Callable]:
        vsd: Dict[str, Callable] = {"val": self.executor.forward}
        if self.use_ema:
            vsd["val_ema"] = self.ema_executor.forward
        return vsd

    def state_dict(self) -> dict:
        res = {
            "state_dict": self.executor.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.use_ema:
            res["state_dict_ema"] = self.ema_executor.model.state_dict()

        return res


def train(
    args,
    device,
    trainer,
    train_step,
    train_loader,
    lr_scheduler,
    logger_res,
    timeout_handler,
    prof=-1,
    step=0,
    epoch=None,
    writer=None,
    model=None,
):
    
    interrupted = False
    
    end = time.time()

    data_iter = enumerate(train_loader)

    module_t, profiler_t = get_tools(model,args)

    mean = (
        torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255])
        .to(f'{args.device}')
        .view(1, 3, 1, 1)
    )
    std = (
        torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255])
        .to(f'{args.device}')
        .view(1, 3, 1, 1)
    )

    for i, (input, target) in data_iter:
        
        os.environ["BATCH_IDX"] = str(i)
        if not args.PrefetchedWrapper:
            if args.device == "cpu" and args.FP64:
                input, target = input.to(torch.float64).cpu(), target.cpu()
            else:
                input = input.float()
                input, target = input.to(device), target.to(device)

        bs = input.size(0)

        if args.mlperf_mode:
            lr = lr_scheduler.get_lr()
        else:
            lr = lr_scheduler(i)
        
        if writer is not None:
            writer.add_scalar("Train/lr", lr, step+i)
        
        input = input.float()
        input = input.sub_(mean).div_(std)
        data_time = time.time() - end

        if args.device == "cpu" and args.FP64 and args.PrefetchedWrapper:
            input, target = input.to(torch.float64).cpu(), target.cpu()

        if args.mlperf_mode:
            loss, fp_time, bp_time, grad_time = train_step(
                input, target, lr_schedule=lr_scheduler, step=step + i)
        else:
            loss, fp_time, bp_time, grad_time = train_step(
                input, target, step=step + i)


        with torch.no_grad():
            if torch.distributed.is_initialized():
                reduced_loss = utils.reduce_tensor(loss.detach())
            else:
                reduced_loss = loss.detach()

        reduced_loss_item = reduced_loss.item()
        it_time = time.time() - end

        if writer is not None:
            writer.add_scalar("Train/Loss", reduced_loss, step+i)
        
        logger_res.log(
            step=(epoch, step + i),
            data = {"rank":args.local_rank,
                    "train.loss":loss.item(), 
                    "train.ips":utils.calc_ips(bs, it_time),
                    "data.shape":[3,224,224],
                    "train.lr":lr,
                    "train.data_time":data_time,
                    "train.compute_time":it_time - data_time,
                    "train.fp_time":fp_time,
                    "train.bp_time":bp_time,
                    "train.grad_time":grad_time,
                    "train.scale":trainer.scale_value,
                    },
            verbosity = Verbosity.DEFAULT,
        )
        logger_res.flush()

        end = time.time()
        
        if prof > 0 and (i + 1 >= prof):
            time.sleep(5)
            break

        t_step(module_t,profiler_t)
        
    t_stop(module_t,profiler_t)
    return interrupted


def validate(args, device, infer_fn, val_loader,logger_res, prof=-1, with_loss=True, topk=5,epoch=None,step=0):
    top1 = log.AverageMeter()
    # switch to evaluate mode

    end = time.time()

    data_iter = enumerate(val_loader)
    mean = (
        torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).to(f'{device}')
        .view(1, 3, 1, 1)
    )
    std = (
        torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).to(f'{device}')
        .view(1, 3, 1, 1)
    )

    for i, (input, target) in data_iter:
        if not args.PrefetchedWrapper:
            input, target = input.to(device), target.to(device)
            input = input.float()

        bs = input.size(0)
        
        input = input.float()
        input = input.sub_(mean).div_(std)
        data_time = time.time() - end
        if with_loss:
            loss, output = infer_fn(input, target)
        else:
            output = infer_fn(input)

        with torch.no_grad():
            precs = utils.accuracy(output.data, target, topk=(1, topk))

            if torch.distributed.is_initialized():
                if with_loss:
                    reduced_loss = utils.reduce_tensor(loss.detach())
                precs = map(utils.reduce_tensor, precs)
            else:
                if with_loss:
                    reduced_loss = loss.detach()

        precs = map(lambda t: t.item(), precs)
        infer_result = {f"top{k}": (p, bs) for k, p in zip((1, topk), precs)}

        if with_loss:
            infer_result["loss"] = (reduced_loss.item(), bs)

        it_time = time.time() - end

        top1.record(infer_result["top1"][0], bs)

        logger_res.log(
            step=(epoch,step),
            data = {
                "val.loss":loss.item(),
                "val.ips":utils.calc_ips(bs, it_time),
                "val.top1":infer_result["top1"][0],
                    },
            verbosity = Verbosity.DEFAULT,
        )
        logger_res.flush()

        end = time.time()
        if (prof > 0) and (i + 1 >= prof):
            time.sleep(5)
            break

    return top1.get_val()


# Train loop {{{
def train_loop(
    args,
    device,
    trainer: Trainer,
    lr_scheduler,
    train_loader,
    train_loader_len,
    val_loader,
    logger_res,
    best_prec1=0,
    start_epoch=0,
    end_epoch=0,
    early_stopping_patience=-1,
    prof=-1,
    skip_training=False,
    skip_validation=False,
    save_checkpoints=True,
    checkpoint_dir="./",
    checkpoint_filename="checkpoint.pth.tar",
    keep_last_n_checkpoints=0,
    topk=5,
    check_f=None,
    check_b=None,
    writer=None,
):
    
    checkpointer = utils.Checkpointer(
        last_filename=checkpoint_filename,
        checkpoint_dir=checkpoint_dir,
        keep_last_n=keep_last_n_checkpoints,
    )

    training_step = trainer.train_step

    prec1 = -1

    if early_stopping_patience > 0:
        epochs_since_improvement = 0

    print(f"RUNNING EPOCHS FROM {start_epoch} TO {end_epoch}")
    with utils.TimeoutHandler(args=args) as timeout_handler:
        interrupted = False
        for epoch in range(start_epoch, end_epoch):
            if args.early_stop >=0:
                if (epoch >=args.early_stop):
                    print(f'early stop at epoch {epoch}')
                    break
            

            if not skip_training:

                data_iter = train_loader

                trainer.train()
                interrupted = train(
                    args,
                    device,
                    trainer,
                    training_step,
                    data_iter,
                    lr_scheduler if args.mlperf_mode else lambda i: lr_scheduler(trainer.optimizer, i, epoch),
                    logger_res,
                    timeout_handler,
                    prof=prof,
                    epoch=epoch,
                    step=epoch * train_loader_len if prof == -1 else epoch * prof,
                    writer=writer,
                    model=trainer.executor.model,
                )

            if not skip_validation:
                val_loader_len = len(val_loader)
                trainer.eval()
                for k, infer_fn in trainer.validation_steps().items():
                    data_iter = val_loader

                    step_prec1, _ = validate(
                        args,
                        device,
                        infer_fn,
                        data_iter,
                        logger_res,
                        prof=prof,
                        topk=topk,
                        epoch=epoch,
                        step=epoch * val_loader_len if prof == -1 else epoch * prof,
                    )

                    if k == "val":
                        prec1 = step_prec1
                best_epoch = epoch
                if prec1 > best_prec1:
                    is_best = True
                    best_prec1 = prec1
                    best_epoch = epoch
                else:
                    is_best = False
                if writer is not None:
                    writer.add_scalar('Val/acc', step_prec1, epoch)
                
                logger_res.log(
                    step=(epoch,0),
                    data = {
                        "summary.epoch":epoch,
                        "summary.metric":prec1,
                        "summary.best_epoch":best_epoch,
                        "summary.best_metric":best_prec1,
                            },
                    verbosity = Verbosity.DEFAULT,
                )
                logger_res.flush()
                
            else:
                is_best = False
                best_prec1 = 0


            if save_checkpoints and (
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ):
                checkpoint_state = {
                    "epoch": epoch + 1,
                    "best_prec1": best_prec1,
                    **trainer.state_dict(),
                }
                checkpointer.save_checkpoint(
                    checkpoint_state,
                    is_best,
                    filename=f"checkpoint_{epoch:04}.pth.tar",
                )

            if early_stopping_patience > 0:
                if not is_best:
                    epochs_since_improvement += 1
                else:
                    epochs_since_improvement = 0
                if epochs_since_improvement >= early_stopping_patience:
                    break
            
            if interrupted:
                break

# }}}
