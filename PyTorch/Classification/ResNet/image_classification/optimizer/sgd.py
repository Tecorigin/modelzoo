import torch
from torch.optim.optimizer import required

def get_sgd_optimizer(
    args,parameters, lr, momentum, weight_decay, nesterov=False, bn_weight_decay=False
):
    if bn_weight_decay:
        print(" ! Weight decay applied to BN parameters ")
        params = [v for n, v in parameters]
    else:
        print(" ! Weight decay NOT applied to BN parameters ")
        if args.arch == 'fused_resnet50':
            params = [v for n, v in parameters]
        else:

            bn_params = [v for n, v in parameters if "bn" in n]
            rest_params = [v for n, v in parameters if not "bn" in n]
            params = [
                {"params": bn_params, "weight_decay": 0},
                {"params": rest_params, "weight_decay": weight_decay},
            ]


    if args.fused_optimizer :
        if args.arch == "fused_resnet50":
            optimizer = FusedResnet50SGD(
                params, lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov,
                bn_weight_decay=bn_weight_decay)
        else :
            import torch_sdaa
            optimizer = torch_sdaa.optim.FusedSGD(
                params, lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov
            )
    else:
        optimizer = torch.optim.SGD(
            params, lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov
        )

    return optimizer

class FusedResnet50SGD(torch.optim.Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)}, \: f(\theta)
                \text{ (objective)}, \: \lambda \text{ (weight decay)},                          \\
            &\hspace{13mm} \:\mu \text{ (momentum)}, \:\tau \text{ (dampening)},\:nesterov\\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}\textbf{if} \: \mu \neq 0                                               \\
            &\hspace{10mm}\textbf{if} \: t > 1                                                   \\
            &\hspace{15mm} \textbf{b}_t \leftarrow \mu \textbf{b}_{t-1} + (1-\tau) g_t           \\
            &\hspace{10mm}\textbf{else}                                                          \\
            &\hspace{15mm} \textbf{b}_t \leftarrow g_t                                           \\
            &\hspace{10mm}\textbf{if} \: nesterov                                                \\
            &\hspace{15mm} g_t \leftarrow g_{t-1} + \mu \textbf{b}_t                             \\
            &\hspace{10mm}\textbf{else}                                                   \\[-1.ex]
            &\hspace{15mm} g_t  \leftarrow  \textbf{b}_t                                         \\
            &\hspace{5mm}\theta_t \leftarrow \theta_{t-1} - \gamma g_t                    \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, bn_weight_decay=True):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        self.bn_weight_decay = bn_weight_decay
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(FusedResnet50SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FusedResnet50SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
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
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']
            if self.bn_weight_decay == True:
                params_with_grad = []
                d_p_list = []
                momentum_buffer_list = []
                for p in group['params']:
                    if p.grad is not None:
                        params_with_grad.append(p)
                        d_p_list.append(p.grad)

                        state = self.state[p]
                        if 'momentum_buffer' not in state:
                            momentum_buffer_list.append(None)
                        else:
                            momentum_buffer_list.append(state['momentum_buffer'])
                is_first = False
                for i, param in enumerate(params_with_grad):
                    buf = momentum_buffer_list[i]
                    if buf is None:
                        buf = torch.clone(d_p_list[i]).detach()
                        momentum_buffer_list[i] = buf
                        is_first = True

                torch.ops.sdaa.fused_sgd(d_p_list, params_with_grad, momentum_buffer_list, weight_decay,
                                        momentum, lr, dampening, nesterov, is_first)


                # update momentum_buffers in state
                for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                    state = self.state[p]
                    state['momentum_buffer'] = momentum_buffer
            else:

                params_with_grad = []
                d_p_list = []
                use_momentum_buffer_list = []

                bn_params_with_grad = []
                bn_d_p_list = []
                bn_use_momentum_buffer_list = []

                momentum_buffer_list = []
                d_p_list_bak = []
                params_with_grad_bak = []
                total_conv_weight_size = (int)(51005952 / 2 + 1024)
                for p in group['params']:
                    if p.grad is not None:
                        params_with_grad_bak.append(p)
                        params_with_grad.append(p[0:total_conv_weight_size])
                        bn_params_with_grad.append(p[total_conv_weight_size:-1])
                        d_p_list.append(p.grad[0:total_conv_weight_size])
                        bn_d_p_list.append(p.grad[total_conv_weight_size:-1])
                        d_p_list_bak.append(p.grad)
                        state = self.state[p]
                        if 'momentum_buffer' not in state:
                            momentum_buffer_list.append(None)
                        else:
                            momentum_buffer_list.append(state['momentum_buffer'])


                is_first = False
                for i, param in enumerate(params_with_grad_bak):
                    buf = momentum_buffer_list[i]
                    if buf is None:
                        buf = torch.clone(d_p_list_bak[i]).detach()
                        momentum_buffer_list[i] = buf
                        is_first = True
                    use_momentum_buffer_list.append(buf[0:total_conv_weight_size])
                    bn_use_momentum_buffer_list.append(buf[total_conv_weight_size:-1])



                torch.ops.sdaa.fused_sgd(d_p_list, params_with_grad, use_momentum_buffer_list, weight_decay,
                                        momentum, lr, dampening, nesterov, is_first)

                torch.ops.sdaa.fused_sgd(bn_d_p_list, bn_params_with_grad, bn_use_momentum_buffer_list, 0.0,
                                        momentum, lr, dampening, nesterov, is_first)

                # update momentum_buffers in state
                for p, momentum_buffer in zip(params_with_grad_bak, momentum_buffer_list):
                    state = self.state[p]
                    state['momentum_buffer'] = momentum_buffer


        return loss
