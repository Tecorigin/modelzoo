# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch
import torch_sdaa

from mmagic.models.editors import BasicVSRPlusPlusNet


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-sdaa due to limited RAM.')
def test_basicvsr_plusplus_cpu():
    """Test BasicVSR++."""

    # cpu
    model = BasicVSRPlusPlusNet(
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=True,
        spynet_pretrained=None,
        cpu_cache_length=100)
    input_tensor = torch.rand(1, 5, 3, 64, 64)
    output = model(input_tensor)
    assert output.shape == (1, 5, 3, 256, 256)

    # with cpu_cache (no effect on cpu)
    model = BasicVSRPlusPlusNet(
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=True,
        spynet_pretrained=None,
        cpu_cache_length=3)
    output = model(input_tensor)
    assert output.shape == (1, 5, 3, 256, 256)

    with pytest.raises(AssertionError):
        # The height and width of inputs should be at least 64
        input_tensor = torch.rand(1, 5, 3, 61, 61)
        model(input_tensor)

    # output has the same size as input
    model = BasicVSRPlusPlusNet(
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=False,
        spynet_pretrained=None,
        cpu_cache_length=100)
    input_tensor = torch.rand(1, 5, 3, 256, 256)
    output = model(input_tensor)
    assert output.shape == (1, 5, 3, 256, 256)


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-sdaa due to limited RAM.')
def test_basicvsr_plusplus_cuda():
    # gpu
    if torch.sdaa.is_available():
        model = BasicVSRPlusPlusNet(
            mid_channels=64,
            num_blocks=7,
            is_low_res_input=True,
            spynet_pretrained=None,
            cpu_cache_length=100).sdaa()
        input_tensor = torch.rand(1, 5, 3, 64, 64).sdaa()
        output = model(input_tensor)
        assert output.shape == (1, 5, 3, 256, 256)

        # with cpu_cache
        model = BasicVSRPlusPlusNet(
            mid_channels=64,
            num_blocks=7,
            is_low_res_input=True,
            spynet_pretrained=None,
            cpu_cache_length=3).sdaa()
        output = model(input_tensor)
        assert output.shape == (1, 5, 3, 256, 256)

        with pytest.raises(AssertionError):
            # The height and width of inputs should be at least 64
            input_tensor = torch.rand(1, 5, 3, 61, 61).sdaa()
            model(input_tensor)

        # output has the same size as input
        model = BasicVSRPlusPlusNet(
            mid_channels=64,
            num_blocks=7,
            is_low_res_input=False,
            spynet_pretrained=None,
            cpu_cache_length=100).sdaa()
        input_tensor = torch.rand(1, 5, 3, 256, 256).sdaa()
        output = model(input_tensor)
        assert output.shape == (1, 5, 3, 256, 256)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
