# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.
import loguru
from torch import nn

from resnet import ResNet, Bottleneck


def resnest50(**kwargs):
    model = ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=32,
        avg_down=True,
        avd=True,
        avd_first=False,
        **kwargs,
    )
    return model


def resnest101(**kwargs):
    model = ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=64,
        avg_down=True,
        avd=True,
        avd_first=False,
        **kwargs,
    )
    return model


def resnest200(**kwargs):
    model = ResNet(
        Bottleneck,
        [3, 24, 36, 3],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=64,
        avg_down=True,
        avd=True,
        avd_first=False,
        **kwargs,
    )
    return model


def resnest269(**kwargs):
    model = ResNet(
        Bottleneck,
        [3, 30, 48, 8],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=64,
        avg_down=True,
        avd=True,
        avd_first=False,
        **kwargs,
    )
    return model


def resnest50_fast(**kwargs):
    model = ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=32,
        avg_down=True,
        avd=True,
        avd_first=True,
        **kwargs,
    )
    return model


def resnest101_fast(**kwargs):
    model = ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=64,
        avg_down=True,
        avd=True,
        avd_first=True,
        **kwargs,
    )
    return model


class ResNeSt:
    @classmethod
    def create_pretrained(cls, model_name, **kwargs) -> nn.Module:
        model = None
        if model_name == "resnest50":
            model = resnest50(**kwargs)
        elif model_name == "resnest101":
            model = resnest101(**kwargs)
        elif model_name == "resnest200":
            model = resnest200(**kwargs)
        elif model_name == "resnest269":
            model = resnest269(**kwargs)
        elif model_name == "resnest50_fast":
            model = resnest50_fast(**kwargs)
        elif model_name == "resnest101_fast":
            model = resnest101_fast(**kwargs)
        else:
            raise NotImplementedError
        loguru.logger.info(f"使用 {model_name} 模型")
        return model
