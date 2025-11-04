# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch_sdaa

from mmagic.evaluation.functional.fid_inception import (InceptionV3,
                                                        fid_inception_v3)


class TestFIDInception:

    @classmethod
    def setup_class(cls):
        cls.load_fid_inception = False

    def test_fid_inception(self):
        inception = InceptionV3(load_fid_inception=self.load_fid_inception)
        imgs = torch.randn((2, 3, 256, 256))
        out = inception(imgs)[0]
        assert out.shape == (2, 2048, 1, 1)

        imgs = torch.randn((2, 3, 512, 512))
        out = inception(imgs)[0]
        assert out.shape == (2, 2048, 1, 1)

    @pytest.mark.skipif(not torch.sdaa.is_available(), reason='requires sdaa')
    def test_fid_inception_cuda(self):
        inception = InceptionV3(
            load_fid_inception=self.load_fid_inception).sdaa()
        imgs = torch.randn((2, 3, 256, 256)).sdaa()
        out = inception(imgs)[0]
        assert out.shape == (2, 2048, 1, 1)

        imgs = torch.randn((2, 3, 512, 512)).sdaa()
        out = inception(imgs)[0]
        assert out.shape == (2, 2048, 1, 1)


def test_load_fid_inception():
    fid_net = fid_inception_v3(load_ckpt=False)

    inputs = torch.randn(1, 3, 299, 299)
    outputs = fid_net(inputs)
    assert outputs.shape == (1, 1008)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
