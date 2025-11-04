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

import logging
import numpy as np
from utils.precision_standard import diff1, diff2, diff3, diff4, mae
from utils.precision_standard import mape, diff3_max, diff3_mean, diff_acc
import tvm


# 用途
# 1. 比较两个输出值的diff误差
def get_diff(config, eval_data, base_data):
    eval_data = np.to_array(eval_data)
    base_data = np.to_array(base_data)

    # 测试平均相对误差 diff1
    if "diff1" in config:
        diff1_result = diff1(eval_data, base_data)
        logging.info(f"{diff1_result}")

    # 均方相对误差开方 diff2
    if "diff2" in config:
        diff2_result = diff2(eval_data, base_data)
        logging.info(f"{diff2_result}")

    # 根据数量级选择相对误差或绝对误差 diff3
    if "diff3" in config:
        diff3_result = diff3(eval_data, base_data)
        logging.info(f"{diff3_result}")

    # 根据数量级选择单点最大相对误差或单点最大绝对误差 diff3_max
    if "diff3_max" in config:
        diff3_max_result = diff3_max(eval_data, base_data)
        logging.info(f"{diff3_max_result}")

    # 度量误差有偏性 diff4
    if "diff4_max" in config:
        diff4_p1, diff4_p2, diff4_n = diff4(eval_data, base_data)
        logging.info(f"{diff4_p1}, {diff4_p2}, {diff4_n}")

    # 平均绝对误差
    if "mae" in config:
        mae_result = mae(eval_data, base_data)
        logging.info(f"{mae_result}")

    # mape
    if "mape" in config:
        mape_result = mape(eval_data, base_data)
        logging.info(f"{mape_result}")


def assert_diff2(sdaa_res, cpu_res):
    # iteration number
    dim_0_size = 0
    if isinstance(sdaa_res[0], np.ndarray):
        logging.debug(f"sdaa_res is [array(iter1), array(iter2), ...], ... ] format")
        eval_data = sdaa_res
        base_data = cpu_res
        dim_0_size = len(eval_data)
    else:
        logging.debug(f"sdaa_res is [[array(output1), array(output2), ...], ... ] format")
        eval_data = np.asarray(sdaa_res)
        base_data = np.asarray(cpu_res)
    if dim_0_size > 0:
        diff2_list = []
        for i in range(dim_0_size):
            diff2_list.append(diff2(eval_data[i], base_data[i]))
        diff2_min = min(diff2_list)
        diff2_max = max(diff2_list)
        diff2_avg = sum(diff2_list) / len(diff2_list)
        diff2_result = diff2_max
        logging.debug(f"Compare difference using diff2, MIN, MAX, AVG is: \
                      {diff2_min:0.12f}, {diff2_max:0.12f}, {diff2_avg:0.12f}")
    else:
        diff2_result = diff2(eval_data, base_data)
    logging.info(f"Compare difference using diff2: {diff2_result:.12f}")


def assert_diff_acc(sdaa_res, cpu_res):
    eval_data = np.asarray(sdaa_res)
    base_data = np.asarray(cpu_res)
    diff_acc_result = diff_acc(eval_data, base_data)
    logging.info(f"Compare difference using diff_acc: {diff_acc_result:.12f}")


def assert_result(sdaa_res, cpu_res, golden_res=None, input_fp32=False, rtol=1e-5, atol=1e-5):
    if golden_res is None:
        tvm.testing.assert_allclose(sdaa_res, cpu_res, rtol=rtol, atol=atol)
    else:
        diff3_max_gold = diff3_max(cpu_res, golden_res, th=1e-6 if input_fp32 else 1e-4)
        diff3_max_sw = diff3_max(sdaa_res, golden_res, th=1e-6 if input_fp32 else 1e-4)
        logging.debug(f"diff3_max_sw vs diff3_max_gold is {diff3_max_sw} vs {diff3_max_gold}")
        assert diff3_max_sw <= diff3_max_gold * 10

        diff3_mean_gold = diff3_mean(cpu_res, golden_res, th=1e-6 if input_fp32 else 1e-4)
        diff3_mean_sw = diff3_mean(sdaa_res, golden_res, th=1e-6 if input_fp32 else 1e-4)
        assert diff3_mean_sw <= diff3_mean_gold * 10


# if __name__ == "__main__":
#     get_diff()
