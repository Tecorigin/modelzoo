# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch
import torch_sdaa

from mmagic.models.editors.stylegan1 import StyleGAN1Discriminator
from mmagic.utils import register_all_modules

register_all_modules()


class TestStyleGANv1Disc:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(in_size=64)

    @pytest.mark.skipif(not torch.sdaa.is_available(), reason='requires sdaa')
    def test_stylegan1_disc_cuda(self):
        d = StyleGAN1Discriminator(**self.default_cfg).sdaa()
        img = torch.randn((2, 3, 64, 64)).sdaa()
        score = d(img)
        assert score.shape == (2, 1)

    @pytest.mark.skipif(
        'win' in platform.system().lower() and 'cu' in torch.__version__,
        reason='skip on windows-sdaa due to limited RAM.')
    def test_stylegan1_disc_cpu(self):
        d = StyleGAN1Discriminator(**self.default_cfg)
        img = torch.randn((2, 3, 64, 64))
        score = d(img)
        assert score.shape == (2, 1)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
