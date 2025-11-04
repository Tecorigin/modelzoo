import numpy as np
import torch

def get_rmsprop_optimizer(
    parameters, lr, alpha, weight_decay, momentum, eps, bn_weight_decay=False
):
    bn_params = [v for n, v in parameters if "bn" in n]
    rest_params = [v for n, v in parameters if not "bn" in n]

    params = [
        {"params": bn_params, "weight_decay": weight_decay if bn_weight_decay else 0},
        {"params": rest_params, "weight_decay": weight_decay},
    ]

    optimizer = torch.optim.RMSprop(
        params,
        lr=lr,
        alpha=alpha,
        weight_decay=weight_decay,
        momentum=momentum,
        eps=eps,
    )

    return optimizer