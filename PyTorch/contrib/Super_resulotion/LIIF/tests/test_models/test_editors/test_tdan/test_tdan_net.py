# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch_sdaa

from mmagic.models.editors import TDANNet


def test_tdan_net():
    """Test TDANNet."""

    # gpu (DCN is available only on GPU)
    if torch.sdaa.is_available():
        tdan = TDANNet().sdaa()
        input_tensor = torch.rand(1, 5, 3, 64, 64).sdaa()

        output = tdan(input_tensor)
        assert len(output) == 2  # (1) HR center + (2) aligned LRs
        assert output[0].shape == (1, 3, 256, 256)  # HR center frame
        assert output[1].shape == (1, 5, 3, 64, 64)  # aligned LRs


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
