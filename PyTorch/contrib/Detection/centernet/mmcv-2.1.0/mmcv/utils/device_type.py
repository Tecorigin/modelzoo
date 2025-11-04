# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.device import (is_sdaa_available, is_mlu_available,
                             is_mps_available, is_npu_available)

IS_MLU_AVAILABLE = is_mlu_available()
IS_MPS_AVAILABLE = is_mps_available()
IS_SDAA_AVAILABLE = is_sdaa_available()
IS_NPU_AVAILABLE = is_npu_available()
