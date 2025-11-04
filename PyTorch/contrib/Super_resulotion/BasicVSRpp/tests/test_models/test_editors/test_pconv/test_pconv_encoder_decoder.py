# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch_sdaa

from mmagic.models.editors import PConvEncoderDecoder


def test_pconv_encdec():
    pconv_enc_cfg = dict(type='PConvEncoder')
    pconv_dec_cfg = dict(type='PConvDecoder')

    if torch.sdaa.is_available():
        pconv_encdec = PConvEncoderDecoder(pconv_enc_cfg, pconv_dec_cfg)
        pconv_encdec.init_weights()
        pconv_encdec.sdaa()
        x = torch.randn((1, 3, 256, 256)).sdaa()
        mask = torch.ones_like(x)
        mask[..., 50:150, 100:250] = 1.
        res, updated_mask = pconv_encdec(x, mask)
        assert res.shape == (1, 3, 256, 256)
        assert mask.shape == (1, 3, 256, 256)

        with pytest.raises(TypeError):
            pconv_encdec.init_weights(pretrained=dict(igccc=8989))


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
