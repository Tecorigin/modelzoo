# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
# from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss # 彻底禁用底层库

from mmdet.registry import MODELS
from .accuracy import accuracy
from .utils import weight_reduce_loss


def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    纯 Python 实现，不依赖任何底层 C++ 扩展，兼容所有硬件。
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    # Actually, pt here denotes (1 - pt) in the Focal Loss paper
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    # Thus it's pt.pow(gamma) rather than (1 - pt).pow(gamma)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    
    # 使用 F.binary_cross_entropy_with_logits 保证数值稳定性
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
        
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def py_focal_loss_with_prob(pred,
                            target,
                            weight=None,
                            gamma=2.0,
                            alpha=0.25,
                            reduction='mean',
                            avg_factor=None):
    """PyTorch version of `Focal Loss`. Accepts probability as input."""
    if pred.dim() != target.dim():
        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes + 1)
        target = target[:, :num_classes]

    target = target.type_as(pred)
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    r"""A wrapper that ensures target is one-hot encoded before calling py implementation.
    """
    # 【核心修复】：如果 target 是索引(1维)，pred 是(2维)，则必须进行 One-Hot 编码
    if pred.dim() != target.dim():
        num_classes = pred.size(1)
        # 将 target 转为 long 类型以支持 one_hot
        target = target.long() 
        # MMDetection 通常使用 num_classes+1 处理背景或忽略索引，我们取前 num_classes
        target = F.one_hot(target, num_classes=num_classes + 1)
        target = target[:, :num_classes]

    return py_sigmoid_focal_loss(
        pred,
        target,
        weight,
        gamma=gamma,
        alpha=alpha,
        reduction=reduction,
        avg_factor=avg_factor)


@MODELS.register_module()
class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0,
                 activated=False):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_
        """
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        if self.use_sigmoid:
            if self.activated:
                calculate_loss_func = py_focal_loss_with_prob
            else:
                # 【关键修改】：不再判断是否是 CUDA，统一调用处理好了 One-Hot 逻辑的 wrapper
                # 这样可以保证无论在 SDAA、CUDA 还是 CPU 上，都先处理形状，再计算 Loss
                calculate_loss_func = sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError
        return loss_cls


@MODELS.register_module()
class FocalCustomLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 num_classes=-1,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0,
                 activated=False):
        super(FocalCustomLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

        assert self.num_classes != -1
        self.custom_cls_channels = True
        self.custom_activation = True
        self.custom_accuracy = True

    def get_cls_channels(self, num_classes):
        assert num_classes == self.num_classes
        return num_classes

    def get_activation(self, cls_score):
        fine_cls_score = cls_score[:, :self.num_classes]
        score_classes = fine_cls_score.sigmoid()
        return score_classes

    def get_accuracy(self, cls_score, labels):
        fine_cls_score = cls_score[:, :self.num_classes]
        pos_inds = labels < self.num_classes
        acc_classes = accuracy(fine_cls_score[pos_inds], labels[pos_inds])
        acc = dict()
        acc['acc_classes'] = acc_classes
        return acc

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            # 这里的逻辑保持不变，因为 num_classes 明确，且已经有 One-Hot 处理
            num_classes = pred.size(1)
            target = F.one_hot(target, num_classes=num_classes + 1)
            target = target[:, :num_classes]
            calculate_loss_func = py_sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError
        return loss_cls