# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import pytest
import torch
import torch_sdaa

from mmagic.models.editors.biggan import BigGANDiscriminator
from mmagic.registry import MODELS


class TestBigGANDiscriminator(object):

    @classmethod
    def setup_class(cls):
        num_classes = 1000
        cls.default_config = dict(
            type='BigGANDiscriminator',
            input_scale=128,
            num_classes=num_classes,
            base_channels=8)
        cls.x = torch.randn((2, 3, 128, 128))
        cls.label = torch.randint(0, num_classes, (2, ))

    def test_biggan_discriminator(self):
        # test default settings
        d = MODELS.build(self.default_config)
        assert isinstance(d, BigGANDiscriminator)
        y = d(self.x, self.label)
        assert y.shape == (2, 1)

        # test different init types
        cfg = deepcopy(self.default_config)
        cfg.update(dict(init_cfg=dict(type='N02')))
        d = MODELS.build(cfg)
        d.init_weights()
        y = d(self.x, self.label)
        assert y.shape == (2, 1)

        cfg = deepcopy(self.default_config)
        cfg.update(dict(init_cfg=dict(type='xavier')))
        d = MODELS.build(cfg)
        d.init_weights()
        y = d(self.x, self.label)
        assert y.shape == (2, 1)

        cfg = deepcopy(self.default_config)
        cfg.update(dict(init_cfg=dict(type='ortho')))
        g = MODELS.build(cfg)
        g.init_weights()
        y = d(self.x, self.label)
        assert y.shape == (2, 1)

        # test different num_classes
        cfg = deepcopy(self.default_config)
        cfg.update(dict(num_classes=0))
        d = MODELS.build(cfg)
        y = d(self.x, None)
        assert y.shape == (2, 1)

        # test with `with_spectral_norm=False`
        cfg = deepcopy(self.default_config)
        cfg.update(dict(with_spectral_norm=False))
        d = MODELS.build(cfg)
        y = d(self.x, self.label)
        assert y.shape == (2, 1)

        # test torch-sn
        cfg = deepcopy(self.default_config)
        cfg.update(dict(sn_style='torch'))
        d = MODELS.build(cfg)
        y = d(self.x, self.label)
        assert y.shape == (2, 1)

    @pytest.mark.skipif(not torch.sdaa.is_available(), reason='requires sdaa')
    def test_biggan_discriminator_cuda(self):
        # test default settings
        d = MODELS.build(self.default_config).sdaa()
        y = d(self.x.sdaa(), self.label.sdaa())
        assert y.shape == (2, 1)

        # test different init types
        cfg = deepcopy(self.default_config)
        cfg.update(dict(init_cfg=dict(type='N02')))
        d = MODELS.build(cfg).sdaa()
        d.init_weights()
        y = d(self.x.sdaa(), self.label.sdaa())
        assert y.shape == (2, 1)

        cfg = deepcopy(self.default_config)
        cfg.update(dict(init_cfg=dict(type='xavier')))
        d = MODELS.build(cfg).sdaa()
        d.init_weights()
        y = d(self.x.sdaa(), self.label.sdaa())
        assert y.shape == (2, 1)

        cfg = deepcopy(self.default_config)
        cfg.update(dict(init_cfg=dict(type='ortho')))
        g = MODELS.build(cfg)
        g.init_weights()
        y = d(self.x, self.label)
        assert y.shape == (2, 1)

        # test different num_classes
        cfg = deepcopy(self.default_config)
        cfg.update(dict(num_classes=0))
        d = MODELS.build(cfg).sdaa()
        y = d(self.x.sdaa(), None)
        assert y.shape == (2, 1)

        # test with `with_spectral_norm=False`
        cfg = deepcopy(self.default_config)
        cfg.update(dict(with_spectral_norm=False))
        d = MODELS.build(cfg).sdaa()
        y = d(self.x.sdaa(), self.label.sdaa())
        assert y.shape == (2, 1)

        # test torch-sn
        cfg = deepcopy(self.default_config)
        cfg.update(dict(sn_style='torch'))
        d = MODELS.build(cfg).sdaa()
        y = d(self.x.sdaa(), self.label.sdaa())
        assert y.shape == (2, 1)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
