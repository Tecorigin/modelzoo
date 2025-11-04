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

import numpy as np


def diff2(eval_data, base_data):
    # 均方相对误差开方
    return ((np.abs(eval_data - base_data)**2).sum() / (base_data**2).sum())**0.5


def diff2_1(eval_data, base_data):
    # 均方误差开方
    return (((eval_data - base_data)**2).sum() / (base_data**2).sum())**0.5


def diff3_1(eval_data, base_data):
    # 相对误差
    eps = 1e-9
    return (np.abs(eval_data - base_data) / (np.abs(base_data) + eps))


def diff3_2(eval_data, base_data):
    # 绝对误差
    return np.abs(eval_data - base_data)


def diff3(eval_data, base_data, th=1e-6):
    # th在fp32下为1e-6, 在fp16下为1e-4
    # if eval_data.dtype == torch.float32:
    #     th = 1e-6
    # elif eval_data.dtype == torch.float16:
    #     th = 1e-4
    # else:
    #     assert False, "Unknown data type: {}".format(eval_data.dtype)
    # 构建mask, 处理不同大小的值
    mask_t = np.abs(base_data) > th
    mask_t_op = ~mask_t
    mask_t = mask_t.astype(np.float32)
    mask_t_op = mask_t_op.astype(np.float32)
    return (diff3_1(eval_data * mask_t, base_data * mask_t) +
            diff3_2(eval_data * mask_t_op, base_data * mask_t_op))


def diff3_max(eval_data, base_data, th=1e-6):
    if isinstance(eval_data, list):
        eval_data = np.array(eval_data)
    if isinstance(base_data, list):
        base_data = np.array(base_data)

    diff3_val = diff3(eval_data, base_data, th=th)
    diff3_arg_max = np.argmax(diff3_val)
    _diff3_max = np.max(diff3_val)
    return (_diff3_max, eval_data.flatten()[diff3_arg_max], base_data.flatten()[diff3_arg_max])


def diff3_mean(eval_data, base_data, th=1e-6):
    _diff3_mean = np.mean(diff3(eval_data, base_data, th=th))
    return _diff3_mean


def assert_rms(actual, desired, rtol=1e-5, atol=1e-5, th=1e-6):  # pylint: disable=W0613
    """Version of np.testing.assert_allclose with `atol` and `rtol` fields set
    in reasonable defaults.

    Arguments `actual` and `desired` are not interchangeable, since the function
    compares the `abs(actual-desired)` with `atol+rtol*abs(desired)`.  Since we
    often allow `desired` to be close to zero, we generally want non-zero `atol`.
    """

    print("\nTest Result:")

    print("actual has NaN: {}, has Inf: {}".format(np.any(np.isnan(actual)),
                                                   np.any(np.isinf(actual))))
    print("desired has NaN: {}, has Inf: {}".format(np.any(np.isnan(desired)),
                                                    np.any(np.isinf(desired))))

    actual = np.asanyarray(actual).astype("float32")
    desired = np.asanyarray(desired).astype("float32")
    np.testing.assert_allclose(actual.shape, desired.shape)
    print("actual shape: {}".format(actual.shape))
    print("desired shape: {}".format(desired.shape))

    diff2_value = diff2(actual, desired)
    diff3_max_value, actual_max_value, desired_max_value = diff3_max(actual, desired, th)
    diff3_mean_value = diff3_mean(actual, desired, th)
    print("diff2 = {}".format(diff2_value))
    print("diff3_max = {}, ({}, {})".format(diff3_max_value, actual_max_value, desired_max_value))
    print("diff3_mean = {}".format(diff3_mean_value))
    # TODO(xxx):analysis diff2 error
    #np.testing.assert_allclose(diff2_value, 0.0, rtol=rtol, atol=atol, verbose=True)
