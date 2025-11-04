# Copyright (c) OpenMMLab. All rights reserved.

import platform

import pytest
import torch
import torch_sdaa

from mmagic.models import TTSRDiscriminator


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-sdaa due to limited RAM.')
def test_ttsr_dict():
    net = TTSRDiscriminator(in_channels=3, in_size=160)
    # cpu
    inputs = torch.rand((2, 3, 160, 160))
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
