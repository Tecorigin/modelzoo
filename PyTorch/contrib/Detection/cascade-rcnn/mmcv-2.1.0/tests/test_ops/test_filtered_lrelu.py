# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmengine.utils import digit_version
from mmengine.utils.dl_utils.parrots_wrapper import is_rocm_pytorch

from mmcv.ops import filtered_lrelu


class TestFilteredLrelu:

    @classmethod
    def setup_class(cls):
        cls.input_tensor = torch.randn((1, 3, 16, 16), requires_grad=True)
        cls.bias = torch.randn(3, requires_grad=True)
        cls.filter_up = torch.randn((2, 2))
        cls.filter_down = torch.randn((2, 2))

    def test_filtered_lrelu_cpu(self):
        out = filtered_lrelu(self.input_tensor, bias=self.bias)
        assert out.shape == (1, 3, 16, 16)

        out = filtered_lrelu(
            self.input_tensor,
            bias=self.bias,
            filter_up=self.filter_up,
            filter_down=self.filter_down,
            up=2,
            down=2,
            padding=1,
            clamp=0.5)
        assert out.shape == (1, 3, 16, 16)

        # test with different filter_up
        filter_up = torch.randn((4, 4))
        out = filtered_lrelu(
            self.input_tensor,
            bias=self.bias,
            filter_up=filter_up,
            filter_down=self.filter_down,
            up=2,
            down=2,
            padding=2,
            clamp=0.5)
        assert out.shape == (1, 3, 16, 16)

        # test with different filter_down
        filter_down = torch.randn((4, 4))
        out = filtered_lrelu(
            self.input_tensor,
            bias=self.bias,
            filter_up=self.filter_up,
            filter_down=filter_down,
            up=2,
            down=2,
            padding=2,
            clamp=0.5)
        assert out.shape == (1, 3, 16, 16)

        # test with different b
        input_tensor = torch.randn((1, 4, 16, 16), requires_grad=True)
        bias = torch.randn(4, requires_grad=True)
        out = filtered_lrelu(
            input_tensor,
            bias=bias,
            filter_up=self.filter_up,
            filter_down=self.filter_down,
            up=2,
            down=2,
            padding=1,
            clamp=0.5)
        assert out.shape == (1, 4, 16, 16)

        # test with different up
        out = filtered_lrelu(
            self.input_tensor,
            bias=self.bias,
            filter_up=self.filter_up,
            filter_down=self.filter_down,
            up=4,
            down=2,
            padding=1,
            clamp=0.5)
        assert out.shape == (1, 3, 32, 32)

        # test with different down
        out = filtered_lrelu(
            self.input_tensor,
            bias=self.bias,
            filter_up=self.filter_up,
            filter_down=self.filter_down,
            up=2,
            down=4,
            padding=1,
            clamp=0.5)
        assert out.shape == (1, 3, 8, 8)

        # test with different gain
        out1 = filtered_lrelu(self.input_tensor, bias=self.bias, gain=0.2)
        out2 = filtered_lrelu(self.input_tensor, bias=self.bias, gain=0.1)
        assert torch.allclose(out1, 2 * out2)

        # test with different slope
        out = filtered_lrelu(self.input_tensor, bias=self.bias, slope=0.2)
        assert out.shape == (1, 3, 16, 16)

        # test with different clamp
        out1 = filtered_lrelu(self.input_tensor, bias=self.bias, clamp=0.2)
        out2 = filtered_lrelu(self.input_tensor, bias=self.bias, clamp=0.1)
        assert out1.max() <= 0.2
        assert out2.max() <= 0.1

        # test with different flip_filter
        out1 = filtered_lrelu(
            self.input_tensor, bias=self.bias, flip_filter=True)
        assert out.shape == (1, 3, 16, 16)

    @pytest.mark.skipif(
        not torch.sdaa.is_available() or is_rocm_pytorch()
        or digit_version(torch.version.sdaa) < digit_version('10.2'),
        reason='requires sdaa>=10.2')
    def test_filtered_lrelu_sdaa(self):
        out = filtered_lrelu(self.input_tensor.sdaa(), bias=self.bias.sdaa())
        assert out.shape == (1, 3, 16, 16)

        out = filtered_lrelu(
            self.input_tensor.sdaa(),
            bias=self.bias.sdaa(),
            filter_up=self.filter_up.sdaa(),
            filter_down=self.filter_down.sdaa(),
            up=2,
            down=2,
            padding=1,
            clamp=0.5)
        assert out.shape == (1, 3, 16, 16)

        # test with different filter_up
        filter_up = torch.randn((4, 4))
        out = filtered_lrelu(
            self.input_tensor.sdaa(),
            bias=self.bias.sdaa(),
            filter_up=filter_up.sdaa(),
            filter_down=self.filter_down.sdaa(),
            up=2,
            down=2,
            padding=2,
            clamp=0.5)
        assert out.shape == (1, 3, 16, 16)

        # test with different filter_down
        filter_down = torch.randn((4, 4))
        out = filtered_lrelu(
            self.input_tensor.sdaa(),
            bias=self.bias.sdaa(),
            filter_up=self.filter_up.sdaa(),
            filter_down=filter_down.sdaa(),
            up=2,
            down=2,
            padding=2,
            clamp=0.5)
        assert out.shape == (1, 3, 16, 16)

        # test with different b
        input_tensor = torch.randn((1, 4, 16, 16), requires_grad=True)
        bias = torch.randn(4, requires_grad=True)
        out = filtered_lrelu(
            input_tensor.sdaa(),
            bias=bias.sdaa(),
            filter_up=self.filter_up.sdaa(),
            filter_down=self.filter_down.sdaa(),
            up=2,
            down=2,
            padding=1,
            clamp=0.5)
        assert out.shape == (1, 4, 16, 16)

        # test with different up
        out = filtered_lrelu(
            self.input_tensor.sdaa(),
            bias=self.bias.sdaa(),
            filter_up=self.filter_up.sdaa(),
            filter_down=self.filter_down.sdaa(),
            up=4,
            down=2,
            padding=1,
            clamp=0.5)
        assert out.shape == (1, 3, 32, 32)

        # test with different down
        out = filtered_lrelu(
            self.input_tensor.sdaa(),
            bias=self.bias.sdaa(),
            filter_up=self.filter_up.sdaa(),
            filter_down=self.filter_down.sdaa(),
            up=2,
            down=4,
            padding=1,
            clamp=0.5)
        assert out.shape == (1, 3, 8, 8)

        # test with different gain
        out1 = filtered_lrelu(
            self.input_tensor.sdaa(), bias=self.bias.sdaa(), gain=0.2)
        out2 = filtered_lrelu(
            self.input_tensor.sdaa(), bias=self.bias.sdaa(), gain=0.1)
        assert torch.allclose(out1, 2 * out2)

        # test with different slope
        out = filtered_lrelu(
            self.input_tensor.sdaa(), bias=self.bias.sdaa(), slope=0.2)
        assert out.shape == (1, 3, 16, 16)

        # test with different clamp
        out1 = filtered_lrelu(
            self.input_tensor.sdaa(), bias=self.bias.sdaa(), clamp=0.2)
        out2 = filtered_lrelu(
            self.input_tensor.sdaa(), bias=self.bias.sdaa(), clamp=0.1)
        assert out1.max() <= 0.2
        assert out2.max() <= 0.1

        # test with different flip_filter
        out1 = filtered_lrelu(
            self.input_tensor.sdaa(), bias=self.bias.sdaa(), flip_filter=True)
        assert out.shape == (1, 3, 16, 16)
