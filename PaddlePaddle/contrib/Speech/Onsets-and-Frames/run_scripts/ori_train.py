import importlib
import os
import warnings
import sys
from pathlib import Path
from argument import config


root_dir = Path(__file__).parent.parent.resolve()
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))

import paddle
paddle.set_device('cpu') # 把加载的数据存在内存里面
import argparse
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
import os
import time
from datetime import datetime
import numpy as np
import paddle.distributed as dist
from paddle.nn.utils import clip_grad_norm_
from paddle.optimizer.lr import StepDecay as StepLR
from paddle.io import DataLoader
from paddle import DataParallel as DP
from visualdl import LogWriter as SummaryWriter
from tqdm import tqdm
from evaluate import evaluate
from onsets_and_frames import *

logger = Logger(
    [
        JSONStreamBackend(Verbosity.VERBOSE, "tcap_dllogger.log"),
    ]
)

dist.init_parallel_env()

def train(logdir, device, nproc_per_node, iterations, resume_iteration, checkpoint_interval, train_on, batch_size, sequence_length,
          model_complexity, amp_on, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, leave_one_out,
          clip_gradient_norm, validation_length, validation_interval):
    
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    train_groups, validation_groups = ['train'], ['validation']

    if leave_one_out is not None:
        all_years = {'2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017'}
        train_groups = list(all_years - {str(leave_one_out)})
        validation_groups = [str(leave_one_out)]

    if train_on == 'MAESTRO':
        dataset = MAESTRO(groups=train_groups, sequence_length=sequence_length)
        validation_dataset = MAESTRO(groups=validation_groups, sequence_length=sequence_length)
    else:
        dataset = MAPS(groups=['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2'], sequence_length=sequence_length)
        validation_dataset = MAPS(groups=['ENSTDkAm', 'ENSTDkCl'], sequence_length=validation_length)

    loader = DataLoader(dataset, batch_size = batch_size, shuffle=True, drop_last=True)
    paddle.set_device(DEFAULT_DEVICE)
    scheduler = StepLR(learning_rate = learning_rate, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)
    model = OnsetsAndFrames(N_MELS, MAX_MIDI - MIN_MIDI + 1, model_complexity)
    if resume_iteration is None:
        optimizer = paddle.optimizer.Adam(parameters = model.parameters(), learning_rate = scheduler)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'model-{resume_iteration}.pdparams')
        model.set_state_dict(paddle.load(model_path))
        optimizer = paddle.optimizer.Adam(parameters = model.parameters(), learning_rate = scheduler)
        if os.path.isfile((file:=os.path.join(logdir, 'last-optimizer-state.pdopt'))):
            optimizer.set_state_dict(paddle.load(file))
            logger.info("load optimizer state at {}".format(file))

    if not (rank:=dist.get_rank()):
        summary(model)
    model = DP(model)
    scaler = paddle.amp.GradScaler()
    if not rank:
        loop = tqdm(range(resume_iteration + 1, iterations + 1))
    else:
        loop = range(resume_iteration + 1, iterations + 1)
    for i, batch in zip(loop, cycle(loader)):
        for j in batch:
            try:
                batch[j] = paddle.to_tensor(batch[j],place=DEFAULT_DEVICE)
            except:
                pass
        
        with paddle.amp.auto_cast(amp_on,custom_black_list=["conv2d","einsum"]):
            predictions, losses = model._layers.run_on_batch(batch) # 向前传播 决速步骤之二
            loss = sum(losses.values()) # total loss
            avg_loss = loss/losses.values().__len__() # avg loss
            
        loss = scaler.scale(loss) # loss 缩放，乘以系数 loss_scaling
        loss.backward()           # 反向传播 决速步骤之一
        scaler.step(optimizer)      # 更新参数
        scaler.update()             # 基于动态 loss_scaling 策略更新 loss_scaling 系数
        optimizer.clear_grad()
        scheduler.step()

        if clip_gradient_norm:
            clip_grad_norm_(model._layers.parameters(), clip_gradient_norm)

        logger_dict = dict()
        # visualdl
        for key, value in {'loss/total_loss': loss, "loss/avg_loss": avg_loss, **losses}.items():
            logger_dict["train." + key.replace('/','.')] = value.item()
            if not rank:
                writer.add_scalar(tag = key, value = value.item(), step = i)
        # file logger
        logger.log(step=i, data=logger_dict)

        if i % validation_interval == 0 and not rank:
            model.eval()
            with paddle.no_grad():
                for key, value in evaluate(validation_dataset, model).items():
                    writer.add_scalar(tag = 'validation/' + key.replace(' ', '_'), value = np.mean(value), step = i)
            model.train()

        if i % checkpoint_interval == 0 and not rank:
            paddle.save(model.state_dict(), os.path.join(logdir, f'model-{i}.pdparams'))
            paddle.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pdopt'))

if __name__ == "__main__":
    cmd = config()
    train(**vars(cmd))
