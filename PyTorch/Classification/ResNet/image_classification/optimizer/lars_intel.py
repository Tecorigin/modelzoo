# Adapted to tecorigin hardware

import torch
from torch.optim.optimizer import Optimizer
from typing import Dict, Iterable, Optional, Callable, Tuple
try:
    import torch_sdaa
    from torch_sdaa.models import fused_resnet50_buffer_size, fused_res50_weight_size, generate_resnet50_dir, compute_offset
except:
    print('import torch_sdaa failed')
from torch import nn
try:
    from torch_sdaa.optim import FusedIntelLARS
except ImportError:
    FusedIntelLARS = None
# FusedIntelLARS = None
"""
    We recommend using create_optimizer_lars and setting bn_bias_separately=True
    instead of using class Lars directly, which helps LARS skip parameters
    in BatchNormalization and bias, and has better performance in general.
    Polynomial Warmup learning rate decay is also helpful for better performance in general.
"""
def create_optimizer_fused_resnet50_lars(model, lr, momentum, weight_decay, bn_bias_separately, epsilon):
    if bn_bias_separately:

        if FusedIntelLARS == None:
            optimizer = FusedResnet50Lars(model.parameters(),
                            lr=lr,
                            momentum=momentum,
                            weight_decay=weight_decay,
                            epsilon=epsilon)
        else:
            optimizer = FusedIntelLARS(model.parameters(),
                            lr=lr,
                            momentum=momentum,
                            weight_decay=weight_decay,
                            epsilon=epsilon)
    else:
        exit(0)
    return optimizer
class FusedResnet50Lars(Optimizer):
    r"""Implements the LARS optimizer from `"Large batch training of convolutional networks"
    <https://arxiv.org/pdf/1708.03888.pdf>`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate
        momentum (float, optional): momentum factor (default: 0)
        eeta (float, optional): LARS coefficient as used in the paper (default: 1e-3)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            lr=1e-3,
            momentum=0,
            eeta=1e-3,
            weight_decay=0,
            epsilon=0.0
    ) -> None:
        # print(params)
        # import pdb; pdb.set_trace()
        if not isinstance(lr, float) or lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if eeta <= 0 or eeta > 1:
            raise ValueError("Invalid eeta value: {}".format(eeta))
        if epsilon < 0:
            raise ValueError("Invalid epsilon value: {}".format(epsilon))
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay, eeta=eeta, epsilon=epsilon, lars=True)

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eeta = group['eeta']
            lr = group['lr']
            # lars = group['lars']
            eps = group['epsilon']
            total_conv_weight_size = (int)(51005952 / 2)
            fc_bias_weigit_size = (int)(1000)
            bn_weight_size = (int)(212480 / 4)
            for p in group['params']:
                normal_weight = p[0:total_conv_weight_size]
                normal_weight_grad = p.grad[0:total_conv_weight_size]
                bias_weight = p[total_conv_weight_size:total_conv_weight_size+fc_bias_weigit_size]
                bias_weight_grad = p.grad[total_conv_weight_size:total_conv_weight_size+fc_bias_weigit_size]
                bn_weight = p[total_conv_weight_size+1024:total_conv_weight_size+1024+bn_weight_size]
                bn_weight_grad = p.grad[total_conv_weight_size+1024:total_conv_weight_size+1024+bn_weight_size]

                #### compute normal weight ####
                start_offset = 0
                end_offset = 3*7*7*64
                def compute_normal_weight(start_, end_, name_):
                    decayed_grad = normal_weight_grad
                    scaled_lr = lr
                    w_norm = torch.norm(normal_weight[start_:end_])
                    g_norm = torch.norm(normal_weight_grad[start_:end_])
                    trust_ratio = torch.where(
                        w_norm > 0 and g_norm > 0,
                        eeta * w_norm / (g_norm + weight_decay * w_norm + eps),
                        torch.ones_like(w_norm)
                    )
                    scaled_lr *= trust_ratio.item()
                    if weight_decay != 0:
                        decayed_grad[start_:end_] = decayed_grad[start_:end_].add(normal_weight[start_:end_], alpha=weight_decay)
                    if momentum != 0:
                        param_state = self.state[p]
                        if name_ not in param_state:
                            buf = param_state[name_] = torch.clone(
                                decayed_grad[start_:end_]).detach()
                        else:
                            buf = param_state[name_]
                            buf.mul_(momentum).add_(decayed_grad[start_:end_])
                        decayed_grad = buf
                    normal_weight[start_:end_].add_(decayed_grad, alpha=-scaled_lr)
                compute_normal_weight(start_offset, end_offset, "first_conv1")
                end_offset = end_offset + 64
                for k,v in fused_res50_weight_size.items():
                    start_offset, end_offset = compute_offset(start_offset, end_offset, v)
                    compute_normal_weight(start_offset, end_offset, k)
                start_offset, end_offset = compute_offset(start_offset, end_offset, 1000*2048)
                compute_normal_weight(start_offset, end_offset, "last_fc")
                assert(total_conv_weight_size == end_offset)
                # decayed_grad = normal_weight_grad
                # scaled_lr = lr
                # w_norm = torch.norm(normal_weight)
                # g_norm = torch.norm(normal_weight_grad)
                # trust_ratio = torch.where(
                #     w_norm > 0 and g_norm > 0,
                #     eeta * w_norm / (g_norm + weight_decay * w_norm + eps),
                #     torch.ones_like(w_norm)
                # )
                # scaled_lr *= trust_ratio.item()
                # if weight_decay != 0:
                #     decayed_grad = decayed_grad.add(normal_weight, alpha=weight_decay)

                # if momentum != 0:
                #     param_state = self.state[p]
                #     if 'normal_momentum_buffer' not in param_state:
                #         buf = param_state['normal_momentum_buffer'] = torch.clone(
                #             decayed_grad).detach()
                #     else:
                #         buf = param_state['normal_momentum_buffer']
                #         buf.mul_(momentum).add_(decayed_grad)
                #     decayed_grad = buf
                # normal_weight.add_(decayed_grad, alpha=-scaled_lr)

                #### Compute bias ####
                decayed_grad = bias_weight_grad
                scaled_lr = lr
                if momentum != 0:
                    param_state = self.state[p]
                    if 'bias_momentum_buffer' not in param_state:
                        buf = param_state['bias_momentum_buffer'] = torch.clone(
                            decayed_grad).detach()
                    else:
                        buf = param_state['bias_momentum_buffer']
                        buf.mul_(momentum).add_(decayed_grad)
                    decayed_grad = buf
                bias_weight.add_(decayed_grad, alpha=-scaled_lr)

                #### Compute bn weight ####
                decayed_grad = bn_weight_grad
                scaled_lr = lr
                if momentum != 0:
                    param_state = self.state[p]
                    if 'bn_momentum_buffer' not in param_state:
                        buf = param_state['bn_momentum_buffer'] = torch.clone(
                            decayed_grad).detach()
                    else:
                        buf = param_state['bn_momentum_buffer']
                        buf.mul_(momentum).add_(decayed_grad)
                    decayed_grad = buf
                bn_weight.add_(decayed_grad, alpha=-scaled_lr)

        return loss
def create_optimizer_lars(model, lr, momentum, weight_decay, bn_bias_separately, epsilon):
    if bn_bias_separately:
        optimizer = Lars([
            dict(params=get_common_parameters(model, exclude_func=get_norm_bias_parameters)),
            dict(params=get_norm_parameters(model), weight_decay=0, lars=False),
            dict(params=get_bias_parameters(model, exclude_func=get_norm_parameters), lars=False)],
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            epsilon=epsilon)
    else:
        optimizer = Lars(model.parameters(),
                         lr=lr,
                         momentum=momentum,
                         weight_decay=weight_decay,
                         epsilon=epsilon)
    return optimizer


class Lars(Optimizer):
    r"""Implements the LARS optimizer from `"Large batch training of convolutional networks"
    <https://arxiv.org/pdf/1708.03888.pdf>`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate
        momentum (float, optional): momentum factor (default: 0)
        eeta (float, optional): LARS coefficient as used in the paper (default: 1e-3)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            lr=1e-3,
            momentum=0,
            eeta=1e-3,
            weight_decay=0,
            epsilon=0.0
    ) -> None:
        if not isinstance(lr, float) or lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if eeta <= 0 or eeta > 1:
            raise ValueError("Invalid eeta value: {}".format(eeta))
        if epsilon < 0:
            raise ValueError("Invalid epsilon value: {}".format(epsilon))
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay,
                        eeta=eeta, epsilon=epsilon, lars=True)

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eeta = group['eeta']
            lr = group['lr']
            lars = group['lars']
            eps = group['epsilon']

            for index_p, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                decayed_grad = p.grad
                scaled_lr = lr
                if lars:
                    w_norm = torch.norm(p)
                    g_norm = torch.norm(p.grad)
                    trust_ratio = torch.where(
                        w_norm > 0 and g_norm > 0,
                        eeta * w_norm / (g_norm + weight_decay * w_norm + eps),
                        torch.ones_like(w_norm)
                    )
                    scaled_lr *= trust_ratio.item()
                    if weight_decay != 0:
                        decayed_grad = decayed_grad.add(p, alpha=weight_decay)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            decayed_grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(decayed_grad)
                    decayed_grad = buf

                p.add_(decayed_grad, alpha=-scaled_lr)
        return loss


"""
    Functions which help to skip bias and BatchNorm
"""
BN_CLS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def get_parameters_from_cls(module, cls_):
    def get_members_fn(m):
        if isinstance(m, cls_):
            return m._parameters.items()
        else:
            return dict()

    named_parameters = module._named_members(get_members_fn=get_members_fn)
    for name, param in named_parameters:
        yield param


def get_norm_parameters(module):
    return get_parameters_from_cls(module, (nn.LayerNorm, *BN_CLS))


def get_bias_parameters(module, exclude_func=None):
    excluded_parameters = set()
    if exclude_func is not None:
        for param in exclude_func(module):
            excluded_parameters.add(param)
    for name, param in module.named_parameters():
        if param not in excluded_parameters and 'bias' in name:
            yield param


def get_norm_bias_parameters(module):
    for param in get_norm_parameters(module):
        yield param
    for param in get_bias_parameters(module, exclude_func=get_norm_parameters):
        yield param


def get_common_parameters(module, exclude_func=None):
    excluded_parameters = set()
    if exclude_func is not None:
        for param in exclude_func(module):
            excluded_parameters.add(param)
    for name, param in module.named_parameters():
        if param not in excluded_parameters:
            yield param
