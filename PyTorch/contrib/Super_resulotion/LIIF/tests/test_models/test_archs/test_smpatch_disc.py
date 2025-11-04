# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch_sdaa

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

register_all_modules()


def test_smpatch_discriminator():
    # color, BN
    cfg = dict(
        type='SoftMaskPatchDiscriminator',
        in_channels=3,
        base_channels=64,
        num_conv=3,
        with_spectral_norm=True)
    net = MODELS.build(cfg)
    # cpu
    input_shape = (1, 3, 64, 64)
    img = torch.rand(input_shape)
    output = net(img)
    assert output.shape == (1, 1, 6, 6)
    # gpu
    if torch.sdaa.is_available():
        net = net.sdaa()
        output = net(img.sdaa())
        assert output.shape == (1, 1, 6, 6)

    # gray, IN
    cfg = dict(
        type='SoftMaskPatchDiscriminator',
        in_channels=1,
        base_channels=64,
        num_conv=3,
        with_spectral_norm=True)
    net = MODELS.build(cfg)
    # cpu
    input_shape = (1, 1, 64, 64)
    img = torch.rand(input_shape)
    output = net(img)
    assert output.shape == (1, 1, 6, 6)
    # gpu
    if torch.sdaa.is_available():
        net = net.sdaa()
        output = net(img.sdaa())
        assert output.shape == (1, 1, 6, 6)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
