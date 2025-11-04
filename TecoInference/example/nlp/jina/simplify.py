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

import onnx
import onnxsim
from onnxconverter_common import float16

import argparse

def str2bool(v):
    """
    将命令行输入的str转换为布尔值
    :param v: str值
    :return:
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--onnx', type=str, default='', help='onnx path')
parser.add_argument('--save_name', type=str, default='', help='onnx path')
opt = parser.parse_args()

if __name__ == "__main__":
    model_onnx = onnx.load(opt.onnx)  # load onnx model

    # 简化模型
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, 'assert check failed'

    fp32_save_name = opt.save_name.replace('.onnx', '_dyn_fp32.onnx')
    fp16_save_name = opt.save_name.replace('.onnx', '_dyn_fp16.onnx')

    # 检查
    onnx.checker.check_model(model_onnx)

    onnx.save(model_onnx, fp32_save_name)
    # 转换数据类型为float16
    model_onnx = float16.convert_float_to_float16(model_onnx)
    onnx.save(model_onnx, fp16_save_name)