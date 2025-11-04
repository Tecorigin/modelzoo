import numpy as np
import torch

def get_adam_optimizer(args,parameters, lr,weight_decay,bn_weight_decay):
    bn_params = [v for n, v in parameters if "bn" in n]
    rest_params = [v for n, v in parameters if not "bn" in n]

    params = [
        {"params": bn_params, "weight_decay": weight_decay if bn_weight_decay else 0},
        {"params": rest_params, "weight_decay": weight_decay},
    ]
    if args.fused_optimizer :

        import torch_sdaa
        optimizer = torch_sdaa.optim.FusedAdam(
            params,
            lr,
        )
    else:
        optimizer = torch.optim.Adam(
            params,
            lr,
            # weight_decay=weight_decay,
        )

    return optimizer








