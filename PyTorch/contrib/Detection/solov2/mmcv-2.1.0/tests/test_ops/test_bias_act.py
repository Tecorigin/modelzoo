# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcv.ops import bias_act
from mmcv.ops.bias_act import EasyDict

_USING_PARROTS = True
try:
    from parrots.autograd import gradcheck
except ImportError:
    from torch.autograd import gradcheck, gradgradcheck
    _USING_PARROTS = False


class TestBiasAct:

    @classmethod
    def setup_class(cls):
        cls.input_tensor = torch.randn((1, 3), requires_grad=True)
        cls.bias = torch.randn(3, requires_grad=True)

    def test_bias_act_cpu(self):
        out = bias_act(self.input_tensor, self.bias)
        assert out.shape == (1, 3)

        # test with different dim
        input_tensor = torch.randn((1, 1, 3), requires_grad=True)
        bias = torch.randn(3, requires_grad=True)
        out = bias_act(input_tensor, bias, dim=2)
        assert out.shape == (1, 1, 3)

        # test with different act
        out = bias_act(self.input_tensor, self.bias, act='relu')
        assert out.shape == (1, 3)
        out = bias_act(self.input_tensor, self.bias, act='lrelu')
        assert out.shape == (1, 3)
        out = bias_act(self.input_tensor, self.bias, act='tanh')
        assert out.shape == (1, 3)
        out = bias_act(self.input_tensor, self.bias, act='sigmoid')
        assert out.shape == (1, 3)
        out = bias_act(self.input_tensor, self.bias, act='elu')
        assert out.shape == (1, 3)
        out = bias_act(self.input_tensor, self.bias, act='selu')
        assert out.shape == (1, 3)
        out = bias_act(self.input_tensor, self.bias, act='softplus')
        assert out.shape == (1, 3)
        out = bias_act(self.input_tensor, self.bias, act='swish')
        assert out.shape == (1, 3)

        # test with different alpha
        out = bias_act(self.input_tensor, self.bias, act='lrelu', alpha=0.1)
        assert out.shape == (1, 3)

        # test with different gain
        out1 = bias_act(self.input_tensor, self.bias, act='lrelu', gain=0.2)
        out2 = bias_act(self.input_tensor, self.bias, act='lrelu', gain=0.1)
        assert torch.allclose(out1, out2 * 2)

        # test with different clamp
        out1 = bias_act(self.input_tensor, self.bias, act='lrelu', clamp=0.5)
        out2 = bias_act(self.input_tensor, self.bias, act='lrelu', clamp=0.2)
        assert out1.max() <= 0.5
        assert out2.max() <= 0.5

    @pytest.mark.skipif(not torch.sdaa.is_available(), reason='requires sdaa')
    def test_bias_act_sdaa(self):
        if _USING_PARROTS:
            gradcheck(
                bias_act, (self.input_tensor.sdaa(), self.bias.sdaa()),
                delta=1e-4,
                pt_atol=1e-3)
        else:
            gradcheck(
                bias_act, (self.input_tensor.sdaa(), self.bias.sdaa()),
                eps=1e-4,
                atol=1e-3)

            gradgradcheck(
                bias_act, (self.input_tensor.sdaa(), self.bias.sdaa()),
                eps=1e-4,
                atol=1e-3)

        out = bias_act(self.input_tensor.sdaa(), self.bias.sdaa())
        assert out.shape == (1, 3)

        # test with different dim
        input_tensor = torch.randn((1, 1, 3), requires_grad=True).sdaa()
        bias = torch.randn(3, requires_grad=True).sdaa()
        out = bias_act(input_tensor, bias, dim=2)
        assert out.shape == (1, 1, 3)

        # test with different act
        out = bias_act(self.input_tensor.sdaa(), self.bias.sdaa(), act='relu')
        assert out.shape == (1, 3)

        out = bias_act(self.input_tensor.sdaa(), self.bias.sdaa(), act='lrelu')
        assert out.shape == (1, 3)
        out = bias_act(self.input_tensor.sdaa(), self.bias.sdaa(), act='tanh')
        assert out.shape == (1, 3)
        out = bias_act(
            self.input_tensor.sdaa(), self.bias.sdaa(), act='sigmoid')
        assert out.shape == (1, 3)
        out = bias_act(self.input_tensor.sdaa(), self.bias.sdaa(), act='elu')
        assert out.shape == (1, 3)
        out = bias_act(self.input_tensor.sdaa(), self.bias.sdaa(), act='selu')
        assert out.shape == (1, 3)
        out = bias_act(
            self.input_tensor.sdaa(), self.bias.sdaa(), act='softplus')
        assert out.shape == (1, 3)
        out = bias_act(self.input_tensor.sdaa(), self.bias.sdaa(), act='swish')
        assert out.shape == (1, 3)

        # test with different alpha
        out = bias_act(
            self.input_tensor.sdaa(), self.bias.sdaa(), act='lrelu', alpha=0.1)
        assert out.shape == (1, 3)

        # test with different gain
        out1 = bias_act(
            self.input_tensor.sdaa(), self.bias.sdaa(), act='lrelu', gain=0.2)
        out2 = bias_act(
            self.input_tensor.sdaa(), self.bias.sdaa(), act='lrelu', gain=0.1)
        assert torch.allclose(out1, out2 * 2)

        # test with different clamp
        out1 = bias_act(
            self.input_tensor.sdaa(), self.bias.sdaa(), act='lrelu', clamp=0.5)
        out2 = bias_act(
            self.input_tensor.sdaa(), self.bias.sdaa(), act='lrelu', clamp=0.2)
        assert out1.max() <= 0.5
        assert out2.max() <= 0.5

    def test_easy_dict(self):
        easy_dict = EasyDict(
            func=lambda x, **_: x,
            def_alpha=0,
            def_gain=1,
            sdaa_idx=1,
            ref='',
            has_2nd_grad=False)
        _ = easy_dict.def_alpha
        easy_dict.def_alpha = 1
        del easy_dict.def_alpha
