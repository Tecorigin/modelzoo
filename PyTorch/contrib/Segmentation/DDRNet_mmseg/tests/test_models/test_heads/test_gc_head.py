# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch_sdaa

from mmseg.models.decode_heads import GCHead
from .utils import to_sdaa


def test_gc_head():
    head = GCHead(in_channels=4, channels=4, num_classes=19)
    assert len(head.convs) == 2
    assert hasattr(head, 'gc_block')
    inputs = [torch.randn(1, 4, 23, 23)]
    if torch.sdaa.is_available():
        head, inputs = to_sdaa(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 23, 23)
