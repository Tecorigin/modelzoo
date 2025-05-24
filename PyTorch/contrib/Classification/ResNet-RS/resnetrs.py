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
"""Pytorch Resnet_RS

This file contains pytorch implementation of Resnet_RS architecture from paper
"Revisiting ResNets: Improved Training and Scaling Strategies"
(https://arxiv.org/pdf/2103.07579.pdf)

"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from base import StemBlock, Bottleneck, Downsample

PRETRAINED_MODELS = ["resnetrs50", "resnetrs101", "resnetrs152", "resnetrs200"]

PRETRAINED_URLS = {
    "resnetrs50": "https://github.com/nachiket273/pytorch_resnet_rs/releases/download/v.0.0.1/resnetrs50_c578f2df.pth",
    "resnetrs101": "https://github.com/nachiket273/pytorch_resnet_rs/releases/download/v.0.0.1/resnetrs101_7c6d6621.pth",
    "resnetrs152": "https://github.com/nachiket273/pytorch_resnet_rs/releases/download/v.0.0.1/resnetrs152_3c858ed0.pth",
    "resnetrs200": "https://github.com/nachiket273/pytorch_resnet_rs/releases/download/v.0.0.1/resnetrs200_fddd5b5f.pth",
}

DEFAULT_CFG = {
    "in_ch": 3,
    "num_classes": 1000,
    "stem_width": 32,
    "down_kernel_size": 1,
    "actn": partial(nn.ReLU, inplace=True),
    "norm_layer": nn.BatchNorm2d,
    "zero_init_last_bn": True,
    "seblock": True,
    "reduction_ratio": 0.25,
    "dropout_ratio": 0.25,
    "conv1": "conv1.conv1.0",
    "stochastic_depth_rate": 0.0,
    "classifier": "fc",
}


def adjust_conv_weights(weights, in_ch=3):
    _, ch, _, _ = weights.shape
    if ch == in_ch:
        return weights

    wtype = weights.dtype
    weights = weights.to(torch.float32)

    if in_ch == 1 and ch == 3:
        # Sum the weights across the input channels.
        weights = weights.sum(dim=1, keepdim=True)
    elif in_ch != 3 and ch == 3:
        new_weights = torch.repeat_interleave(weights, int(math.ceil(in_ch / 3)), dim=1)
        weights = new_weights[:, :ch, :, :]
        weights *= 3.0 / in_ch
    else:
        raise NotImplementedError("Conversion not implemented.")
    return weights.to(wtype)


def get_pretrained_weights(url, cfg, num_classes=1000, in_ch=3, map_location="cpu", progress=False, check_hash=False):
    state_dict = load_state_dict_from_url(url, progress=progress, map_location=map_location, check_hash=check_hash)

    conv1_name = cfg["conv1"]
    classifier_name = cfg["classifier"]
    model_in_ch = cfg["in_ch"]
    model_num_classes = cfg["num_classes"]

    if in_ch != model_in_ch:
        try:
            state_dict[conv1_name + ".weight"] = adjust_conv_weights(state_dict[conv1_name + ".weight"], in_ch)
        except NotImplementedError:
            del state_dict[conv1_name + ".weight"]
            cfg["strict"] = False

    if num_classes != model_num_classes:
        del state_dict[classifier_name + ".weight"]
        del state_dict[classifier_name + ".bias"]
        cfg["strict"] = False

    return state_dict


class Resnet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        in_ch=3,
        stem_width=64,
        down_kernel_size=1,
        actn=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        seblock=True,
        reduction_ratio=0.25,
        dropout_ratio=0.0,
        stochastic_depth_ratio=0.0,
        zero_init_last_bn=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.norm_layer = norm_layer
        self.actn = actn
        self.dropout_ratio = float(dropout_ratio)
        self.stochastic_depth_ratio = stochastic_depth_ratio
        self.zero_init_last_bn = zero_init_last_bn
        self.conv1 = StemBlock(in_ch, stem_width, norm_layer, actn)
        channels = [64, 128, 256, 512]
        self.make_layers(block, layers, channels, stem_width * 2, down_kernel_size, seblock, reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes, bias=True)

    def make_layers(self, block, nlayers, channels, inplanes, kernel_size=1, seblock=True, reduction_ratio=0.25):
        tot_nlayers = sum(nlayers)
        layer_idx = 0
        for idx, (nlayer, channel) in enumerate(zip(nlayers, channels)):
            name = "layer" + str(idx + 1)
            stride = 1 if idx == 0 else 2
            downsample = None
            if stride != 1 or inplanes != channel * block.expansion:
                downsample = Downsample(
                    inplanes, channel * block.expansion, kernel_size=kernel_size, stride=stride, norm_layer=self.norm_layer
                )

            blocks = []
            for layer_idx in range(nlayer):
                downsample = downsample if layer_idx == 0 else None
                stride = stride if layer_idx == 0 else 1
                drop_ratio = self.stochastic_depth_ratio * layer_idx / (tot_nlayers - 1)
                blocks.append(
                    block(
                        inplanes,
                        channel,
                        stride,
                        self.norm_layer,
                        self.actn,
                        downsample,
                        seblock,
                        reduction_ratio,
                        drop_ratio,
                        self.zero_init_last_bn,
                    )
                )

                inplanes = channel * block.expansion
                layer_idx += 1

            self.add_module(*(name, nn.Sequential(*blocks)))

    def init_weights(self):
        for _, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.flatten(1, -1)
        if self.dropout_ratio > 0.0:
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.fc(x)
        return x


class ResnetRS:
    def __init__(self):
        super().__init__()

    @classmethod
    def create_model(
        cls,
        block,
        layers,
        num_classes=1000,
        in_ch=3,
        stem_width=64,
        down_kernel_size=1,
        actn=partial(nn.ReLU, inplace=True),
        norm_layer=nn.BatchNorm2d,
        seblock=True,
        reduction_ratio=0.25,
        dropout_ratio=0.0,
        stochastic_depth_rate=0.0,
        zero_init_last_bn=True,
    ):

        return Resnet(
            block,
            layers,
            num_classes=num_classes,
            in_ch=in_ch,
            stem_width=stem_width,
            down_kernel_size=down_kernel_size,
            actn=actn,
            norm_layer=norm_layer,
            seblock=seblock,
            reduction_ratio=reduction_ratio,
            dropout_ratio=dropout_ratio,
            stochastic_depth_ratio=stochastic_depth_rate,
            zero_init_last_bn=zero_init_last_bn,
        )

    @classmethod
    def list_pretrained(cls):
        return PRETRAINED_MODELS

    @classmethod
    def _is_valid_model_name(cls, name):
        name = name.strip()
        name = name.lower()
        return name in PRETRAINED_MODELS

    @classmethod
    def _get_url(cls, name):
        return PRETRAINED_URLS[name]

    @classmethod
    def _get_default_cfg(cls):
        return DEFAULT_CFG

    @classmethod
    def _get_cfg(cls, name):
        cfg = ResnetRS._get_default_cfg()
        cfg["block"] = Bottleneck
        if name == "resnetrs50":
            cfg["layers"] = [3, 4, 6, 3]
        elif name == "resnetrs101":
            cfg["layers"] = [3, 4, 23, 3]
        elif name == "resnetrs152":
            cfg["layers"] = [3, 8, 36, 3]
        elif name == "resnetrs200":
            cfg["layers"] = [3, 24, 36, 3]
            cfg["stochastic_depth_rate"] = 0.1
        return cfg

    @classmethod
    def create_pretrained(cls, name, in_ch=0, num_classes=0, drop_rate=0.0, need_weights=False):
        if not ResnetRS._is_valid_model_name(name):
            raise ValueError("Available pretrained models: " + ", ".join(PRETRAINED_MODELS))

        cfg = ResnetRS._get_cfg(name)
        in_ch = cfg["in_ch"] if in_ch == 0 else in_ch
        num_classes = cfg["num_classes"] if num_classes == 0 else num_classes

        if drop_rate > 0.0:
            cfg["stochastic_depth_rate"] = drop_rate

        model = Resnet(
            cfg["block"],
            cfg["layers"],
            num_classes=num_classes,
            in_ch=in_ch,
            stem_width=cfg["stem_width"],
            down_kernel_size=cfg["down_kernel_size"],
            actn=cfg["actn"],
            norm_layer=cfg["norm_layer"],
            seblock=cfg["seblock"],
            dropout_ratio=cfg["dropout_ratio"],
            reduction_ratio=cfg["reduction_ratio"],
            stochastic_depth_ratio=cfg["stochastic_depth_rate"],
            zero_init_last_bn=cfg["zero_init_last_bn"],
        )

        if need_weights:
            url = ResnetRS._get_url(name)
            cfg["strict"] = True
            state_dict = get_pretrained_weights(url, cfg, num_classes, in_ch, check_hash=True)
            model.load_state_dict(state_dict, strict=cfg["strict"])

        return model
