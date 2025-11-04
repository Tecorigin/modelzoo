# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted to tecorigin hardware

from __future__ import absolute_import, division, print_function

import time
import paddle
from ppcls.engine.train.utils import update_loss, update_metric, log_info
from tcap_dllogger import  Verbosity
import paddle.distributed as dist
from ppcls.utils import get_tools, t_step, t_stop

def train_epoch(engine, epoch_id, print_batch_step,writer):
    tic = time.time()
    
    module_t, profiler_t = get_tools(engine)
    
    for iter_id, batch in enumerate(engine.train_dataloader):
        
        _data_time = time.time()
        data_time = _data_time -tic
        
        if iter_id >= engine.max_iter:
            break

        if iter_id == 5:
            for key in engine.time_info:
                engine.time_info[key].reset()
        engine.time_info["reader_cost"].update(time.time() - tic)
        if engine.use_dali:
            batch = [
                paddle.to_tensor(batch[0]['data']),
                paddle.to_tensor(batch[0]['label'])
            ]
        batch_size = batch[0].shape[0]
        if not engine.config["Global"].get("use_multilabel", False):
            batch[1] = batch[1].reshape([batch_size, -1])
        engine.global_step += 1

        # image input
        if engine.amp:
            amp_level = engine.config['AMP'].get("level", "O1").upper()
            with paddle.amp.auto_cast(
                    custom_black_list=engine.amp_list_black,
                    custom_white_list=engine.amp_list_white,
                    level=amp_level):
                out = forward(engine, batch)
                loss_dict = engine.train_loss_func(out, batch[1])
        else:
            out = forward(engine, batch)
            loss_dict = engine.train_loss_func(out, batch[1])

        # loss
        loss = loss_dict["loss"] / engine.update_freq
        
        if engine.config["Global"]['visualdl']:
            if paddle.distributed.get_rank() == 0:
                writer.add_scalar('Loss',loss.item(),(epoch_id-1) * engine.max_iter + iter_id)
        
        _fp_time = time.time()
        fp_time = _fp_time - _data_time
        
        # backward & step opt
        if engine.amp:
            scaled = engine.scaler.scale(loss)
            scaled.backward()
            if (iter_id + 1) % engine.update_freq == 0:
                for i in range(len(engine.optimizer)):
                    engine.scaler.minimize(engine.optimizer[i], scaled)
        else:
            loss.backward()
            if (iter_id + 1) % engine.update_freq == 0:
                for i in range(len(engine.optimizer)):
                    engine.optimizer[i].step()

        _bp_time = time.time()
        bp_time = _bp_time - _fp_time
        
        if (iter_id + 1) % engine.update_freq == 0:
            # clear grad
            for i in range(len(engine.optimizer)):
                engine.optimizer[i].clear_grad(set_to_zero=False)
                
            # update ema
            if engine.ema:
                engine.model_ema.update(engine.model)

        _grad_time = time.time()
        grad_time = _grad_time - _bp_time
        
        # below code just for logging
        # update metric_for_logger
        update_metric(engine, out, batch, batch_size)
        # update_loss_for_logger
        update_loss(engine, loss_dict, batch_size)
        
        for i, lr in enumerate(engine.lr_sch):
            lr_value = lr.get_lr()
        
        batch_cost = time.time() - tic
        engine.time_info["batch_cost"].update(batch_cost)

        if iter_id % print_batch_step == 0:
            engine.teco_logger.log(
                step = (epoch_id,engine.global_step),
                data = {"rank":dist.get_rank(),
                        "train.loss":loss.item(), 
                        "train.ips":batch_size/(batch_cost),
                        "data.shape":[batch_size,3,224,224],
                        "train.lr":lr_value,
                        "train.data_time":data_time,
                        "train.compute_time":fp_time+bp_time+grad_time,
                        "train.fp_time":fp_time,
                        "train.bp_time":bp_time,
                        "train.grad_time":grad_time,
                },
                verbosity=Verbosity.DEFAULT,
            )
        
        tic = time.time()
        
        t_step(module_t,profiler_t)
        
        if engine.config["Global"]['prof']>=0 and (iter_id+1) >=engine.config["Global"]['prof']:
            break
        
    t_stop(module_t,profiler_t)
    # step lr(by epoch)
    for i in range(len(engine.lr_sch)):
        engine.lr_sch[i].step()


def forward(engine, batch):
    if not engine.is_rec:
        if engine.config['Global']['device']=='cpu' and engine.config['Global']['FP64']:
            batch[0] = paddle.to_tensor( batch[0], dtype=paddle.float64)
        return engine.model(batch[0])
    else:
        if engine.config['Global']['device']=='cpu' and engine.config['Global']['FP64']:
            batch[0] = batch[0].to('cpu',dtype=paddle.float64)
            batch[1] = batch[1].to('cpu',dtype=paddle.float64)
        return engine.model(batch[0], batch[1])
