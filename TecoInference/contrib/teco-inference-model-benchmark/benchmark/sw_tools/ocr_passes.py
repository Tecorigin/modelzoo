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

import tvm
from tvm import relay
from tvm.relay.op.contrib.tecodnn import (fuse_conv2d_bias_silu, fuse_conv2d_bias_add_relu,
                                          fuse_conv2d_bias_relu, fuse_conv2d_bias,
                                          fuse_conv_bn_add_act, resize2d_cast, fuse_add_batchnorm,
                                          sigmoid_cast_trans)


def run_pass(ir_module, target, passes=None, opt_level=3):
    with tvm.target.Target(target):
        if passes is not None:
            seq = tvm.transform.Sequential(passes)

            with tvm.transform.PassContext(opt_level=opt_level):
                ir_module = seq(ir_module)

    return ir_module


def use_passes(ir_module, load_model_configs=None):
    if load_model_configs is None:
        load_model_configs = {}

    # target = load_model_configs.get("target", "sdaa --libs=tecodnn,tecoblas")
    # opt_level = load_model_configs.get("opt_level", 3)

    passes1 = [
        relay.transform.InferType(),
        relay.transform.SimplifyExpr(),
        relay.transform.CanonicalizeOps(),
        relay.transform.AlterOpLayout(),
        relay.transform.FoldConstant(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.DeadCodeElimination(),
        relay.transform.InferType(),
    ]
    ir_module = fuse_add_batchnorm(ir_module)
    with tvm.target.Target("sdaa"):
        with tvm.transform.PassContext(opt_level=3):
            ir_module = relay.transform.CanonicalizeOps()(ir_module)
            ir_module = relay.transform.SimplifyInference()(ir_module)
            ir_module = relay.transform.FoldScaleAxis()(ir_module)
    ir_module = run_pass(ir_module, "sdaa", passes1)
    with tvm.target.Target("sdaa"):
        with tvm.transform.PassContext(opt_level=3):
            ir_module = relay.transform.ChannelPadding()(ir_module)
            ir_module = relay.transform.LayoutTransform5D()(ir_module)
            ir_module = relay.transform.SimplifyPad()(ir_module)
            ir_module = relay.transform.SimplifyExpr()(ir_module)
    ir_module = fuse_conv_bn_add_act(ir_module)
    ir_module = fuse_conv2d_bias_silu(ir_module)
    ir_module = fuse_conv2d_bias_add_relu(ir_module)
    ir_module = fuse_conv2d_bias_relu(ir_module)
    ir_module = fuse_conv2d_bias(ir_module)
    ir_module = resize2d_cast(ir_module)
    ir_module = sigmoid_cast_trans(ir_module)
    return ir_module
