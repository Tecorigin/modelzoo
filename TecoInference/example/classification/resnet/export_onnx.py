import onnx
import onnxsim		# 用于简化模型
from onnxconverter_common import float16 # 用于将模型转为float16
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torchvision

# init model

resnet = torchvision.models.resnet50(pretrained=True)
resnet.eval()

# init dummy_input
dummy_input = torch.randn(1, 3, 224, 224)

# 静态shape导出
torch.onnx.export(resnet,
                  dummy_input,
                  "resnet.onnx",
                  opset_version=12,         	# ONNX opset版本
                  input_names=['input'],    	# 输入名称
                  output_names=['output'],  	# 输出名称
                  do_constant_folding=True, 	# 是否执行常量折叠优化
                  dynamic_axes=None,	# 是否使用动态shape，不使用默认为None
                 )

# 动态shape导出（推荐）
dynamic_dims = {'input': {0: 'batch', 2: 'height', 3: 'width'},
                'output': {0: 'batch'}}

torch.onnx.export(resnet,
                  dummy_input,
                  "resnet_dyn.onnx",
                  opset_version=12,         	# ONNX opset版本
                  input_names=['input'],    	# 输入名称
                  output_names=['output'],  	# 输出名称
                  do_constant_folding=True, 	# 是否执行常量折叠优化
                  dynamic_axes=dynamic_dims,	# 是否使用动态shape，不使用默认为None
                 )

# 以下动态静态均适用

# Checks
model_onnx = onnx.load("resnet_dyn.onnx")  # load onnx model
onnx.checker.check_model(model_onnx)  # check onnx model

# Simplify
model_onnx, check = onnxsim.simplify(model_onnx)
assert check, 'assert check failed'

# convert_float_to_float16
model_onnx = float16.convert_float_to_float16(model_onnx)
onnx.save(model_onnx, "resnet_float16_dyn.onnx")