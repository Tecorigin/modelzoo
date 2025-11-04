# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch
import torch_sdaa
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.utils.version_utils import digit_version

from mmagic.models.editors.stylegan2 import StyleGAN2Generator
from mmagic.models.losses import GeneratorPathRegularizerComps


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-sdaa due to limited RAM.')
class TestPathRegularizer:

    @classmethod
    def setup_class(cls):
        cls.data_info = dict(generator='generator', num_batches='num_batches')
        cls.gen = StyleGAN2Generator(32, 10, num_mlps=2)

    def test_path_regularizer_cpu(self):
        gen = self.gen

        output_dict = dict(generator=gen, num_batches=2)
        pl = GeneratorPathRegularizerComps(data_info=self.data_info)
        pl_loss = pl(output_dict)
        assert pl_loss > 0

        output_dict = dict(generator=gen, num_batches=2, iteration=3)
        pl = GeneratorPathRegularizerComps(
            data_info=self.data_info, interval=2)
        pl_loss = pl(outputs_dict=output_dict)
        assert pl_loss is None

        with pytest.raises(NotImplementedError):
            _ = pl(asdf=1.)

        with pytest.raises(AssertionError):
            _ = pl(1.)

        with pytest.raises(AssertionError):
            _ = pl(1., 2, outputs_dict=output_dict)

    @pytest.mark.skipif(
        digit_version(TORCH_VERSION) <= digit_version('1.6.0'),
        reason='version limitation')
    @pytest.mark.skipif(not torch.sdaa.is_available(), reason='requires sdaa')
    def test_path_regularizer_cuda(self):
        gen = self.gen.sdaa()

        output_dict = dict(generator=gen, num_batches=2)
        pl = GeneratorPathRegularizerComps(data_info=self.data_info).sdaa()
        pl_loss = pl(output_dict)
        assert pl_loss > 0

        output_dict = dict(generator=gen, num_batches=2, iteration=3)
        pl = GeneratorPathRegularizerComps(
            data_info=self.data_info, interval=2).sdaa()
        pl_loss = pl(outputs_dict=output_dict)
        assert pl_loss is None

        with pytest.raises(NotImplementedError):
            _ = pl(asdf=1.)

        with pytest.raises(AssertionError):
            _ = pl(1.)

        with pytest.raises(AssertionError):
            _ = pl(1., 2, outputs_dict=output_dict)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
