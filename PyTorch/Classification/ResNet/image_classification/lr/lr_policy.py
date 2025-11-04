import math

import numpy as np
import torch
from torch import optim

def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    return _alr


def lr_step_policy(base_lr, steps, decay_factor, warmup_length):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            lr = base_lr
            for s in steps:
                if epoch >= s:
                    lr *= decay_factor
        return lr

    return lr_policy(_lr_fn)

def no_lr_policy(base_lr):
    def _lr_fn(iteration, epoch):
        return base_lr

    return lr_policy(_lr_fn)


def lr_linear_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = base_lr * (1 - (e / es))
        return lr

    return lr_policy(_lr_fn)


def lr_cosine_policy(base_lr, warmup_length, epochs, end_lr=0):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = end_lr + (0.5 * (1 + np.cos(np.pi * e / es)) * (base_lr - end_lr))
        return lr

    return lr_policy(_lr_fn)


def lr_exponential_policy(
    base_lr,
    warmup_length,
    epochs,
    final_multiplier=0.001,
    decay_factor=None,
    decay_step=1,
    logger=None,
):
    """Exponential lr policy. Setting decay factor parameter overrides final_multiplier"""
    es = epochs - warmup_length

    if decay_factor is not None:
        epoch_decay = decay_factor
    else:
        epoch_decay = np.power(
            2, np.log2(final_multiplier) / math.floor(es / decay_step)
        )

    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            lr = base_lr * (epoch_decay ** math.floor(e / decay_step))
        return lr

    return lr_policy(_lr_fn, logger=logger)


class MLPerfLRScheduler:
    '''
    Implements LR schedule according to MLPerf Tensorflow2 reference for Resnet50
    This scheduler needs to be called before optimizer.step()
    '''
    def __init__(self, optimizer, train_epochs, warmup_epochs, steps_per_epoch, base_lr, epoch=None,end_lr=0.0001, power=2.0):

        self.optimizer = optimizer
        self.base_lr =  base_lr
        self.end_lr = end_lr
        self.power = power
        self.train_steps = train_epochs*steps_per_epoch
        self.warmup_steps = warmup_epochs*steps_per_epoch
        self.decay_steps = self.train_steps - self.warmup_steps + 1
        self.current_lr = None
        self.current_step = 0
        self.epoch = epoch
        if self.epoch !=0:
            self.current_step = steps_per_epoch * self.epoch

    def step(self):
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            self.current_lr = self._get_warmup_rate(self.current_step)
        else:
            self.current_lr = self._get_poly_rate(self.current_step)

        self._update_optimizer_lr(self.current_lr)

    def _get_warmup_rate(self, step):

        return self.base_lr*(step/self.warmup_steps)

    def get_lr(self):

        if self.current_step <= self.warmup_steps:
            self.current_lr = self._get_warmup_rate(self.current_step)
        else:
            self.current_lr = self._get_poly_rate(self.current_step)
        return self.current_lr

    def _get_poly_rate(self, step):

        poly_step = step - self.warmup_steps
        poly_rate = (self.base_lr - self.end_lr)*(1-(poly_step/self.decay_steps))**self.power + self.end_lr
        return poly_rate

    def _update_optimizer_lr(self, lr):

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr