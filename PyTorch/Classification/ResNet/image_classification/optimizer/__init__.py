import math

import numpy as np
import torch
from torch import optim
from .adam import get_adam_optimizer
from .rmsprop import get_rmsprop_optimizer
from .sgd import get_sgd_optimizer
from .lars_intel import create_optimizer_fused_resnet50_lars,create_optimizer_lars

def get_optimizer(parameters, lr, args, state=None):
    if args.optimizer == "sgd":
        optimizer = get_sgd_optimizer(
            args,
            parameters,
            lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
            bn_weight_decay=args.bn_weight_decay,
        )
    elif args.optimizer == "rmsprop":
        optimizer = get_rmsprop_optimizer(
            parameters,
            lr,
            alpha=args.rmsprop_alpha,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            eps=args.rmsprop_eps,
            bn_weight_decay=args.bn_weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = get_adam_optimizer(
            args,
            parameters,
            lr,
            weight_decay=args.weight_decay,
            bn_weight_decay=args.bn_weight_decay,
        )
    if not state is None:
        optimizer.load_state_dict(state)

    return optimizer