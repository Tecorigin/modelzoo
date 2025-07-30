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
import torch
import torch.nn as nn
import argparse
from model.ssd import SSDDetector

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='/mnt/nvme/common/user_data/yqw/ssd/mobilenet_v3_ssd320_voc0712.pth', help='onnx path')
    opt = parser.parse_args()
    return opt

opt = parse_opt()

# 初始化修改后的模型
model = SSDDetector()
checkpoint = torch.load(opt.ckpt, map_location=torch.device("cpu"))
model.load_state_dict(checkpoint.pop("model"))
model.eval()

# 创建虚拟输入
dummy_input = torch.randn(1, 3, 320, 320)

# 动态shape导出
dynamic_dims = {'input': {0: 'batch', 2: 'height', 3: 'width'},
                'output': {0: 'batch'}}

# 导出为ONNX格式
torch.onnx.export(model,
                  dummy_input,
                  "ssd_dyn.onnx",
                  opset_version=12,         	# ONNX opset版本
                  input_names=['input'],    	# 输入名称
                  output_names=['output'],  	# 输出名称
                  do_constant_folding=True, 	# 是否执行常量折叠优化
                  dynamic_axes=dynamic_dims)	# 动态shape

# 检查导出的ONNX模型
model_onnx = onnx.load("ssd_dyn.onnx")  # 加载ONNX模型
onnx.checker.check_model(model_onnx)  # 检查ONNX模型

# 简化ONNX模型
model_onnx, check = onnxsim.simplify(model_onnx)
assert check, '简化模型检查失败'

# 转换模型到float16
model_onnx = float16.convert_float_to_float16(model_onnx)
onnx.save(model_onnx, "ssd_dyn_float16.onnx")  # 保存转换后的模型

# 静态shape导出
torch.onnx.export(model,
                  dummy_input,
                  "ssd.onnx",
                  opset_version=12,         	# ONNX opset版本
                  input_names=['input'],    	# 输入名称
                  output_names=['output'],  	# 输出名称
                  do_constant_folding=True, 	# 是否执行常量折叠优化
                  dynamic_axes=None,	# 是否使用动态shape，不使用默认为None
                 )

