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
from typing import Optional

import paddle
from paddle import nn


class ConvNeXtBlock(nn.Layer):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        dilation (int, optional): Dilation factor for the depthwise convolution. Defaults to 1.
        kernel_size (int, optional): Kernel size for the depthwise convolution. Defaults to 7.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to 1e-6.
    """

    def __init__(
            self,
            dim: int,
            intermediate_dim: int,
            dilation: int = 1,
            kernel_size: int = 7,
            layer_scale_init_value: Optional[float] = 1e-6,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            groups=dim,
            dilation=dilation,
            padding=int(dilation * (kernel_size - 1) / 2),
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * paddle.ones(dim), requires_grad=True)
            if layer_scale_init_value is not None and layer_scale_init_value > 0
            else None
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        residual = x

        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class ConvNeXt(nn.Layer):
    """ConvNeXt layers

    Args:
        dim (int): Number of input channels.
        num_layers (int): Number of ConvNeXt layers.
        mlp_factor (int, optional): Factor for the intermediate layer dimensionality. Defaults to 4.
        dilation_cycle (int, optional): Cycle for the dilation factor. Defaults to 4.
        kernel_size (int, optional): Kernel size for the depthwise convolution. Defaults to 7.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to 1e-6.
    """

    def __init__(
            self,
            dim: int,
            num_layers: int = 20,
            mlp_factor: int = 4,
            dilation_cycle: int = 4,
            kernel_size: int = 7,
            layer_scale_init_value: Optional[float] = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.mlp_factor = mlp_factor
        self.dilation_cycle = dilation_cycle
        self.kernel_size = kernel_size
        self.layer_scale_init_value = layer_scale_init_value

        self.layers = nn.LayerList(
            [
                ConvNeXtBlock(
                    dim,
                    dim * mlp_factor,
                    dilation=(2 ** (i % dilation_cycle)),
                    kernel_size=kernel_size,
                    layer_scale_init_value=1e-6,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
