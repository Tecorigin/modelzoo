# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch
import torch_sdaa

from mmagic.models.editors import MaxFeature
from mmagic.registry import MODELS


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-sdaa due to limited RAM.')
def test_max_feature():
    # cpu
    conv2d = MaxFeature(16, 16, filter_type='conv2d')
    x1 = torch.rand(3, 16, 16, 16)
    y1 = conv2d(x1)
    assert y1.shape == (3, 16, 16, 16)
    linear = MaxFeature(16, 16, filter_type='linear')
    x2 = torch.rand(3, 16)
    y2 = linear(x2)
    assert y2.shape == (3, 16)
    # gpu
    if torch.sdaa.is_available():
        x1 = x1.sdaa()
        x2 = x2.sdaa()
        conv2d = conv2d.sdaa()
        linear = linear.sdaa()
        y1 = conv2d(x1)
        assert y1.shape == (3, 16, 16, 16)
        y2 = linear(x2)
        assert y2.shape == (3, 16)
    # filter_type should be conv2d or linear
    with pytest.raises(ValueError):
        MaxFeature(12, 12, filter_type='conv1d')


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-sdaa due to limited RAM.')
def test_light_cnn():
    cfg = dict(type='LightCNN', in_channels=3)
    net = MODELS.build(cfg)
    # cpu
    inputs = torch.rand((2, 3, 128, 128))
    output = net(inputs)
    assert output.shape == (2, 1)
    # gpu
    if torch.sdaa.is_available():
        net = net.sdaa()
        output = net(inputs.sdaa())
        assert output.shape == (2, 1)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
