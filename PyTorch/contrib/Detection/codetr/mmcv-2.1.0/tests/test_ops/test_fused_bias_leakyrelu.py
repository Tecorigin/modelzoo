# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcv.utils import IS_SDAA_AVAILABLE, IS_NPU_AVAILABLE

_USING_PARROTS = True
try:
    from parrots.autograd import gradcheck
except ImportError:
    from torch.autograd import gradcheck, gradgradcheck
    _USING_PARROTS = False


class TestFusedBiasLeakyReLU:

    @classmethod
    def setup_class(cls):
        if not IS_SDAA_AVAILABLE and not IS_NPU_AVAILABLE:
            return
        if IS_SDAA_AVAILABLE:
            cls.input_tensor = torch.randn((2, 2, 2, 2),
                                           requires_grad=True).sdaa()
            cls.bias = torch.zeros(2, requires_grad=True).sdaa()
        elif IS_NPU_AVAILABLE:
            cls.input_tensor = torch.randn((2, 2, 2, 2),
                                           requires_grad=True).npu()
            cls.bias = torch.zeros(2, requires_grad=True).npu()

    @pytest.mark.parametrize('device', [
        pytest.param(
            'sdaa',
            marks=pytest.mark.skipif(
                not IS_SDAA_AVAILABLE, reason='requires SDAA support')),
        pytest.param(
            'npu',
            marks=pytest.mark.skipif(
                not IS_NPU_AVAILABLE, reason='requires NPU support'))
    ])
    def test_gradient(self, device):

        from mmcv.ops import FusedBiasLeakyReLU
        if _USING_PARROTS:
            if IS_SDAA_AVAILABLE:
                gradcheck(
                    FusedBiasLeakyReLU(2).sdaa(),
                    self.input_tensor,
                    delta=1e-4,
                    pt_atol=1e-3)
        else:
            gradcheck(
                FusedBiasLeakyReLU(2).to(device),
                self.input_tensor,
                eps=1e-4,
                atol=1e-3)

    @pytest.mark.parametrize('device', [
        pytest.param(
            'sdaa',
            marks=pytest.mark.skipif(
                not IS_SDAA_AVAILABLE, reason='requires SDAA support')),
        pytest.param(
            'npu',
            marks=pytest.mark.skipif(
                not IS_NPU_AVAILABLE, reason='requires NPU support'))
    ])
    def test_gradgradient(self, device):

        from mmcv.ops import FusedBiasLeakyReLU
        gradgradcheck(
            FusedBiasLeakyReLU(2).to(device),
            self.input_tensor,
            eps=1e-4,
            atol=1e-3)
