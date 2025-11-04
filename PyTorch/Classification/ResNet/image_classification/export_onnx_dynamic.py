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


import sys
import torch
#模型导入
from models.resnet import resnet50
from torchvision.models import resnet18, resnet101
import argparse
from onnxconverter_common import float16
import onnx
def convert(model_name,ckpt):
    #固定onnx模型的输入与batchsize，根据需要判断是否使用AMP模式
    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")

    #实例化模型
    model = get_model(model_name)
    
    if '50' in model_name:
        model = model(pretrained=False)
        state_dict = torch.load(ckpt)

        state_dict = {
                    k[len("module.") :] if k.startswith("module.") else k: v
                    for k, v in state_dict.items()
                }
        remap_fn = model.ngc_checkpoint_remap()
        state_dict = {remap_fn(k): v for k, v in state_dict.items()}
    else:
        model = model()
        state_dict = torch.load(ckpt)

    model.load_state_dict(state_dict)

    
    #导出onnx模型，四个参数分别为模型，模型输入，导出onnx模型名称，input_names（在使用时请与导出时的名称一致）
    torch.onnx.export(model, dummy_input, f"{model_name}_fp32_dynamic.onnx",input_names=['input'],output_names = ['output'],
                      dynamic_axes={'input': {0: 'batch_size', 2 : 'in_width', 3: 'int_height'},
                                    'output':{0: 'batch_size', 2 : 'in_width', 3: 'int_height'}})#verbose=True, input_names=input_names, output_names=output_names

    model = onnx.load(f"{model_name}_fp32_dynamic.onnx")

    model_fp16 = float16.convert_float_to_float16(model)
    
    onnx.save(model_fp16,f"{model_name}_fp16_dynamic.onnx")
    

def get_model(model_name):
    if '18' in model_name:
        model = resnet18
    if '50' in model_name:
        model = resnet50
    if '101' in model_name:
        model = resnet101
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Infer ResNet to onnx')
    parser.add_argument('--model', type=str, default='resnet50', help='model to export')
    parser.add_argument('--ckpt', type=str, default=None, help='path to ckpt file')
    args = parser.parse_args()
    
    model = args.model
    ckpt = args.ckpt
    convert(model,ckpt)