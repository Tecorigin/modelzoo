# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch_sdaa

from mmagic.models.editors.mspie import MSStyleGAN2Discriminator


class TestMSStyleGANv2Disc:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(in_size=64, channel_multiplier=1)

    def test_msstylegan2_disc_cpu(self):
        d = MSStyleGAN2Discriminator(**self.default_cfg)
        img = torch.randn((2, 3, 64, 64))
        score = d(img)
        assert score.shape == (2, 1)

        d = MSStyleGAN2Discriminator(
            with_adaptive_pool=True, **self.default_cfg)
        img = torch.randn((2, 3, 64, 64))
        score = d(img)
        assert score.shape == (2, 1)

    @pytest.mark.skipif(not torch.sdaa.is_available(), reason='requires sdaa')
    def test_msstylegan2_disc_cuda(self):
        d = MSStyleGAN2Discriminator(**self.default_cfg).sdaa()
        img = torch.randn((2, 3, 64, 64)).sdaa()
        score = d(img)
        assert score.shape == (2, 1)

        d = MSStyleGAN2Discriminator(
            with_adaptive_pool=True, **self.default_cfg).sdaa()
        img = torch.randn((2, 3, 64, 64)).sdaa()
        score = d(img)
        assert score.shape == (2, 1)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
