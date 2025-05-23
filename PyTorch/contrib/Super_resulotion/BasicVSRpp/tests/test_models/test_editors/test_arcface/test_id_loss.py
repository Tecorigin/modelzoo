# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch
import torch_sdaa

from mmagic.models import IDLossModel


class TestArcFace:

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
        # test loss model
        id_loss_model = IDLossModel()
        x1 = torch.randn((2, 3, 224, 224))
        x2 = torch.randn((2, 3, 224, 224))
        y, _ = id_loss_model(pred=x1, gt=x2)
        assert y >= 0

    @pytest.mark.skipif(not torch.sdaa.is_available(), reason='requires sdaa')
    def test_arcface_cuda(self):
        # test loss model
        id_loss_model = IDLossModel().sdaa()
        x1 = torch.randn((2, 3, 224, 224)).sdaa()
        x2 = torch.randn((2, 3, 224, 224)).sdaa()
        y, _ = id_loss_model(pred=x1, gt=x2)
        assert y >= 0


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
