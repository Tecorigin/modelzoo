# Copyright (c) OpenMMLab. All rights reserved.
import torch


class TestInfo:

    def test_info(self):
        if not torch.sdaa.is_available():
            return
        from mmcv.ops import get_compiler_version, get_compiling_sdaa_version
        cv = get_compiler_version()
        ccv = get_compiling_sdaa_version()
        assert cv is not None
        assert ccv is not None
