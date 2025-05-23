# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import torch
import torch_sdaa


class ScalarBias(torch.autograd.Function):
    """
    Adds a vector of scalars, used in self-attention mechanism to allow
    the model to optionally attend to this vector instead of the past
    """

    @staticmethod
    def forward(ctx, input, dim, bias_init):
        size = list(input.size())
        size[dim] += 1
        output = input.new(*size).fill_(bias_init)
        output.narrow(dim, 1, size[dim] - 1).copy_(input)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad):
        return grad.narrow(ctx.dim, 1, grad.size(ctx.dim) - 1), None, None


def scalar_bias(input, dim, bias_init=0):
    return ScalarBias.apply(input, dim, bias_init)
