# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch_sdaa

from mmagic.models.editors.lsgan import LSGANDiscriminator
from mmagic.registry import MODELS


class TestLSGANDiscriminator(object):

    @classmethod
    def setup_class(cls):
        cls.x = torch.randn((2, 3, 128, 128))
        cls.default_config = dict(
            type='LSGANDiscriminator', in_channels=3, input_scale=128)

    def test_lsgan_discriminator(self):

        # test default setting with builder
        d = MODELS.build(self.default_config)
        assert isinstance(d, LSGANDiscriminator)
        score = d(self.x)
        assert score.shape == (2, 1)

        # test different input_scale
        config = dict(type='LSGANDiscriminator', in_channels=3, input_scale=64)
        d = MODELS.build(config)
        assert isinstance(d, LSGANDiscriminator)
        x = torch.randn((2, 3, 64, 64))
        score = d(x)
        assert score.shape == (2, 1)

        # test different config
        config = dict(
            type='LSGANDiscriminator',
            in_channels=3,
            input_scale=64,
            out_act_cfg=dict(type='Sigmoid'))
        d = MODELS.build(config)
        assert isinstance(d, LSGANDiscriminator)
        x = torch.randn((2, 3, 64, 64))
        score = d(x)
        assert score.shape == (2, 1)

    @pytest.mark.skipif(not torch.sdaa.is_available(), reason='requires sdaa')
    def test_lsgan_discriminator_cuda(self):

        # test default setting with builder
        d = MODELS.build(self.default_config).sdaa()
        assert isinstance(d, LSGANDiscriminator)
        score = d(self.x.sdaa())
        assert score.shape == (2, 1)

        # test different input_scale
        config = dict(type='LSGANDiscriminator', in_channels=3, input_scale=64)
        d = MODELS.build(config).sdaa()
        assert isinstance(d, LSGANDiscriminator)
        x = torch.randn((2, 3, 64, 64))
        score = d(x.sdaa())
        assert score.shape == (2, 1)

        # test different config
        config = dict(
            type='LSGANDiscriminator',
            in_channels=3,
            input_scale=64,
            out_act_cfg=dict(type='Sigmoid'))
        d = MODELS.build(config).sdaa()
        assert isinstance(d, LSGANDiscriminator)
        x = torch.randn((2, 3, 64, 64))
        score = d(x.sdaa())
        assert score.shape == (2, 1)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
