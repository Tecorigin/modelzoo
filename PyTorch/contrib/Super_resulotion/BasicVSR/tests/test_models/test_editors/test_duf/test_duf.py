# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch_sdaa

from mmagic.models.editors.duf import DynamicUpsamplingFilter


def test_dynamic_upsampling_filter():
    """Test DynamicUpsamplingFilter."""
    with pytest.raises(TypeError):
        # The type of filter_size must be tuple
        DynamicUpsamplingFilter(filter_size=3)
    with pytest.raises(ValueError):
        # The length of filter size must be 2
        DynamicUpsamplingFilter(filter_size=(3, 3, 3))

    duf = DynamicUpsamplingFilter(filter_size=(5, 5))
    x = torch.rand(1, 3, 4, 4)
    filters = torch.rand(1, 25, 16, 4, 4)
    output = duf(x, filters)
    assert output.shape == (1, 48, 4, 4)

    duf = DynamicUpsamplingFilter(filter_size=(3, 3))
    x = torch.rand(1, 3, 4, 4)
    filters = torch.rand(1, 9, 16, 4, 4)
    output = duf(x, filters)
    assert output.shape == (1, 48, 4, 4)

    # gpu (since it has dcn, only supports gpu testing)
    if torch.sdaa.is_available():
        duf = DynamicUpsamplingFilter(filter_size=(3, 3)).sdaa()
        x = torch.rand(1, 3, 4, 4).sdaa()
        filters = torch.rand(1, 9, 16, 4, 4).sdaa()
        output = duf(x, filters)
        assert output.shape == (1, 48, 4, 4)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
