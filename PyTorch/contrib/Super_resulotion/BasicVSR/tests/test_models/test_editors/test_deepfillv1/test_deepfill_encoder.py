# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch_sdaa

from mmagic.models.archs import SimpleGatedConvModule
from mmagic.models.editors import DeepFillEncoder


def test_deepfill_enc():
    encoder = DeepFillEncoder()
    x = torch.randn((2, 5, 256, 256))
    outputs = encoder(x)
    assert isinstance(outputs, dict)
    assert 'out' in outputs
    res = outputs['out']
    assert res.shape == (2, 128, 64, 64)
    assert encoder.enc2.stride == (2, 2)
    assert encoder.enc2.out_channels == 64

    encoder = DeepFillEncoder(encoder_type='stage2_conv')
    x = torch.randn((2, 5, 256, 256))
    outputs = encoder(x)
    assert isinstance(outputs, dict)
    assert 'out' in outputs
    res = outputs['out']
    assert res.shape == (2, 128, 64, 64)
    assert encoder.enc2.out_channels == 32
    assert encoder.enc3.out_channels == 64
    assert encoder.enc4.out_channels == 64

    encoder = DeepFillEncoder(encoder_type='stage2_attention')
    x = torch.randn((2, 5, 256, 256))
    outputs = encoder(x)
    assert isinstance(outputs, dict)
    assert 'out' in outputs
    res = outputs['out']
    assert res.shape == (2, 128, 64, 64)
    assert encoder.enc2.out_channels == 32
    assert encoder.enc3.out_channels == 64
    assert encoder.enc4.out_channels == 128
    if torch.sdaa.is_available():
        encoder = DeepFillEncoder().sdaa()
        x = torch.randn((2, 5, 256, 256)).sdaa()
        outputs = encoder(x)
        assert isinstance(outputs, dict)
        assert 'out' in outputs
        res = outputs['out']
        assert res.shape == (2, 128, 64, 64)
        assert encoder.enc2.stride == (2, 2)
        assert encoder.enc2.out_channels == 64

        encoder = DeepFillEncoder(encoder_type='stage2_conv').sdaa()
        x = torch.randn((2, 5, 256, 256)).sdaa()
        outputs = encoder(x)
        assert isinstance(outputs, dict)
        assert 'out' in outputs
        res = outputs['out']
        assert res.shape == (2, 128, 64, 64)
        assert encoder.enc2.out_channels == 32
        assert encoder.enc3.out_channels == 64
        assert encoder.enc4.out_channels == 64

        encoder = DeepFillEncoder(encoder_type='stage2_attention').sdaa()
        x = torch.randn((2, 5, 256, 256)).sdaa()
        outputs = encoder(x)
        assert isinstance(outputs, dict)
        assert 'out' in outputs
        res = outputs['out']
        assert res.shape == (2, 128, 64, 64)
        assert encoder.enc2.out_channels == 32
        assert encoder.enc3.out_channels == 64
        assert encoder.enc4.out_channels == 128

        encoder = DeepFillEncoder(
            conv_type='gated_conv', channel_factor=0.75).sdaa()
        x = torch.randn((2, 5, 256, 256)).sdaa()
        outputs = encoder(x)
        assert isinstance(outputs, dict)
        assert 'out' in outputs
        res = outputs['out']
        assert res.shape == (2, 96, 64, 64)
        assert isinstance(encoder.enc2, SimpleGatedConvModule)
        assert encoder.enc2.conv.stride == (2, 2)
        assert encoder.enc2.conv.out_channels == 48 * 2


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
