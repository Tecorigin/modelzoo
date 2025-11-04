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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter

from nets_arcface.iresnet import (iresnet18, iresnet34, iresnet50, iresnet100,
                          iresnet200)
from nets_arcface.mobilefacenet import get_mbf
from nets_arcface.mobilenet import get_mobilenet

class Arcface_Head(Module):
    def __init__(self, embedding_size=128, num_classes=10575, s=64., m=0.5):
        super(Arcface_Head, self).__init__()
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine  = F.linear(input, F.normalize(self.weight))
        sine    = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi     = cosine * self.cos_m - sine * self.sin_m
        phi     = torch.where(cosine.float() > self.th, phi.float(), cosine.float() - self.mm)

        one_hot = torch.zeros(cosine.size()).type_as(phi).long()
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output  = (one_hot * phi) + ((1.0 - one_hot) * cosine) 
        output  *= self.s
        return output

class Arcface(nn.Module):
    def __init__(self, num_classes=None, backbone="mobilefacenet", pretrained=False, mode="train"):
        super(Arcface, self).__init__()
        if backbone=="mobilefacenet":
            embedding_size  = 128
            s               = 32
            self.arcface    = get_mbf(embedding_size=embedding_size, pretrained=pretrained)

        elif backbone=="mobilenetv1":
            embedding_size  = 512
            s               = 64
            self.arcface    = get_mobilenet(dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)

        elif backbone=="iresnet18":
            embedding_size  = 512
            s               = 64
            self.arcface    = iresnet18(dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)

        elif backbone=="iresnet34":
            embedding_size  = 512
            s               = 64
            self.arcface    = iresnet34(dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)

        elif backbone=="iresnet50":
            embedding_size  = 512
            s               = 64
            self.arcface    = iresnet50(dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)

        elif backbone=="iresnet100":
            embedding_size  = 512
            s               = 64
            self.arcface    = iresnet100(dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)

        elif backbone=="iresnet200":
            embedding_size  = 512
            s               = 64
            self.arcface    = iresnet200(dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilefacenet, mobilenetv1.'.format(backbone))

        self.mode = mode
        if mode == "train":
            self.head = Arcface_Head(embedding_size=embedding_size, num_classes=num_classes, s=s)

    def forward(self, x, y = None, mode = "predict"):
        x = self.arcface(x)
        x = x.view(x.size()[0], -1)
        x = F.normalize(x)
        if mode == "predict":
            return x
        else:
            x = self.head(x, y)
            return x
