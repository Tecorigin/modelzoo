# Copyright (c) OpenMMLab. All rights reserved.
import platform
from copy import deepcopy

import pytest
import torch
import torch_sdaa

from mmagic.models.editors.arcface.model_irse import Backbone


class TestIRSEModel:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            input_size=224,
            num_layers=50,
            mode='ir',
            drop_ratio=0.4,
            affine=True)

    @pytest.mark.skipif(
        'win' in platform.system().lower() and 'cu' in torch.__version__,
        reason='skip on windows-sdaa due to limited RAM.')
    def test_arcface_cpu(self):
        model = Backbone(**self.default_cfg)
        x = torch.randn((2, 3, 224, 224))
        y = model(x)
        assert y.shape == (2, 512)

        # test different input size
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(input_size=112))
        model = Backbone(**cfg)
        x = torch.randn((2, 3, 112, 112))
        y = model(x)
        assert y.shape == (2, 512)

        # test different num_layers
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(num_layers=50))
        model = Backbone(**cfg)
        x = torch.randn((2, 3, 224, 224))
        y = model(x)
        assert y.shape == (2, 512)

        # test different mode
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(mode='ir_se'))
        model = Backbone(**cfg)
        x = torch.randn((2, 3, 224, 224))
        y = model(x)
        assert y.shape == (2, 512)

        # test different drop ratio
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(drop_ratio=0.8))
        model = Backbone(**cfg)
        x = torch.randn((2, 3, 224, 224))
        y = model(x)
        assert y.shape == (2, 512)

        # test affine=False
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(affine=False))
        model = Backbone(**cfg)
        x = torch.randn((2, 3, 224, 224))
        y = model(x)
        assert y.shape == (2, 512)

    @pytest.mark.skipif(not torch.sdaa.is_available(), reason='requires sdaa')
    def test_arcface_cuda(self):
        model = Backbone(**self.default_cfg).sdaa()
        x = torch.randn((2, 3, 224, 224)).sdaa()
        y = model(x)
        assert y.shape == (2, 512)

        # test different input size
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(input_size=112))
        model = Backbone(**cfg).sdaa()
        x = torch.randn((2, 3, 112, 112)).sdaa()
        y = model(x)
        assert y.shape == (2, 512)

        # test different num_layers
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(num_layers=50))
        model = Backbone(**cfg).sdaa()
        x = torch.randn((2, 3, 224, 224)).sdaa()
        y = model(x)
        assert y.shape == (2, 512)

        # test different mode
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(mode='ir_se'))
        model = Backbone(**cfg).sdaa()
        x = torch.randn((2, 3, 224, 224)).sdaa()
        y = model(x)
        assert y.shape == (2, 512)

        # test different drop ratio
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(drop_ratio=0.8))
        model = Backbone(**cfg).sdaa()
        x = torch.randn((2, 3, 224, 224)).sdaa()
        y = model(x)
        assert y.shape == (2, 512)

        # test affine=False
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(affine=False))
        model = Backbone(**cfg).sdaa()
        x = torch.randn((2, 3, 224, 224)).sdaa()
        y = model(x)
        assert y.shape == (2, 512)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
