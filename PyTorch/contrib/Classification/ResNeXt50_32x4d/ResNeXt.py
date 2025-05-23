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
from typing import List, Union, Tuple

import torch.nn as nn
import torch.nn.functional as F


class Conv2DBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ResNeXtBlock(nn.Module):
    """ResNeXt block with group convolutions"""

    def __init__(self, in_channels, cardinality: int, group_depth, stride):
        super().__init__()
        group_chnls = cardinality * group_depth
        self.conv1 = Conv2DBNReLU(in_channels, group_chnls, 1, stride=1, padding=0)
        self.conv2 = Conv2DBNReLU(group_chnls, group_chnls, 3, stride=stride, padding=1, groups=cardinality)
        self.conv3 = nn.Conv2d(group_chnls, group_chnls * 2, 1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(group_chnls * 2)
        self.short_cut = nn.Sequential(
            nn.Conv2d(in_channels, group_chnls * 2, 1, stride, 0, bias=False), nn.BatchNorm2d(group_chnls * 2)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(self.conv3(out))
        out += self.short_cut(x)
        return F.relu(out)


class ResNeXt(nn.Module):
    """ResNeXt builder"""

    def __init__(self, layers: Union[List[int] | Tuple[int]], cardinality: int, group_depth, num_classes):
        super().__init__()
        self.cardinality = cardinality
        self.channels = 64
        self.conv1 = Conv2DBNReLU(3, self.channels, 7, stride=2, padding=3)
        d1 = group_depth
        self.conv2 = self.__make_layers(d1, layers[0], stride=1)
        d2 = d1 * 2
        self.conv3 = self.__make_layers(d2, layers[1], stride=2)
        d3 = d2 * 2
        self.conv4 = self.__make_layers(d3, layers[2], stride=2)
        d4 = d3 * 2
        self.conv5 = self.__make_layers(d4, layers[3], stride=2)
        self.fc = nn.Linear(self.channels, num_classes)  # 224x224 input size

    def __make_layers(self, d, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResNeXtBlock(self.channels, self.cardinality, d, stride))
            self.channels = self.cardinality * d * 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = F.softmax(self.fc(out), dim=1)
        return out


def create_ResNeXt50_32x4d(num_classes=1000):
    return ResNeXt([3, 4, 6, 3], 50, 4, num_classes)
