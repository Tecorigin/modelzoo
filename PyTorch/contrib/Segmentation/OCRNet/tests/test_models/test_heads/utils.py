# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import ConvModule
from mmengine.utils.dl_utils.parrots_wrapper import SyncBatchNorm


def _conv_has_norm(module, sync_bn):
    for m in module.modules():
        if isinstance(m, ConvModule):
            if not m.with_norm:
                return False
            if sync_bn:
                if not isinstance(m.bn, SyncBatchNorm):
                    return False
    return True


def to_sdaa(module, data):
    module = module.sdaa()
    if isinstance(data, list):
        for i in range(len(data)):
            data[i] = data[i].sdaa()
    return module, data


def list_to_sdaa(data):
    if isinstance(data, list):
        for i in range(len(data)):
            data[i] = list_to_sdaa(data[i])
        return data
    else:
        return data.sdaa()
