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

# 需要处理的问题
# 1. tensor->np.array
# 2. 处理异常值，多数为base中为0的值

eps = 1e-9

lateset_precision_standard = ["diff3_max", "diff3_mean"]


def diff1(eval_data, base_data):
    # 平均相对误差
    return np.abs(eval_data - base_data).sum() / np.abs(base_data).sum()


def diff2(eval_data, base_data):
    # 均方相对误差开方
    return ((np.abs(eval_data - base_data)**2).sum() / (base_data**2).sum())**0.5


def diff_acc(eval_data, base_data):
    # specific metric for mlperf resnet50
    return (eval_data.astype(np.int64) != base_data.astype(np.int64)).sum() / base_data.size


def diff2_1(eval_data, base_data):
    # 均方误差开方
    return (((eval_data - base_data)**2).sum() / (base_data**2).sum())**0.5


def diff3_1(eval_data, base_data):
    # 相对误差
    # eps = 1e-9
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
    return np.max(diff3(eval_data, base_data, th=th))


def diff3_mean(eval_data, base_data, th=1e-6):
    return np.mean(diff3(eval_data, base_data, th=th))


def mae(eval_data, base_data):
    # 平均绝对误差
    return np.mean(np.abs(eval_data - base_data))


def diff4(eval_data, base_data):
    # 误差有偏性度量
    greater = np.greater(eval_data, base_data)
    greater_count = np.count_nonzero(greater)
    less = np.less(eval_data, base_data)
    less_count = np.count_nonzero(less)
    not_equal_count = greater_count + less_count
    if not_equal_count != 0:
        p_greater = greater_count / not_equal_count
        p_less = less_count / not_equal_count
    else:
        p_greater = -1
        p_less = -1

    return p_greater, p_less, not_equal_count


def mape(eval_data, base_data):
    # eps = 1e-9
    # mape
    return np.mean(np.abs((eval_data - base_data) / (abs(base_data) + eps)))
