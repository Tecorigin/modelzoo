# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch_sdaa

from mmseg.models.decode_heads import ANNHead
from .utils import to_sdaa


def test_ann_head():

    inputs = [torch.randn(1, 4, 45, 45), torch.randn(1, 8, 21, 21)]
    head = ANNHead(
        in_channels=[4, 8],
        channels=2,
        num_classes=19,
        in_index=[-2, -1],
        project_channels=8)
    if torch.sdaa.is_available():
        head, inputs = to_sdaa(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 21, 21)
