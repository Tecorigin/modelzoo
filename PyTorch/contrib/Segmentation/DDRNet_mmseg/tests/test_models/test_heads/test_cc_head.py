# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch_sdaa

from mmseg.models.decode_heads import CCHead
from .utils import to_sdaa


def test_cc_head():
    head = CCHead(in_channels=16, channels=8, num_classes=19)
    assert len(head.convs) == 2
    assert hasattr(head, 'cca')
    if not torch.sdaa.is_available():
        pytest.skip('CCHead requires sdaa')
    inputs = [torch.randn(1, 16, 23, 23)]
    head, inputs = to_sdaa(head, inputs)
    outputs = head(inputs)
    assert outputs.shape == (1, head.num_classes, 23, 23)
