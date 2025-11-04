import onnx
import onnxsim		# 用于简化模型
from onnxconverter_common import float16	# 用于将模型转为float16

import paddle
from paddle.vision.models import resnet50

# init model
model = resnet50(pretrained=True)

# 静态shape
input_spec = [
    paddle.static.InputSpec(
        shape=[1, 3, 224, 224], dtype="float32"),
]

# 动态shape
input_spec = [
    paddle.static.InputSpec(
        shape=[-1, 3, -1, -1], dtype="float32"),
]
paddle.onnx.export(model,
                   "resnet.onnx",
                   input_spec=input_spec,
                   opset_version=12)

# 以下动态静态均适用

# Checks
model_onnx = onnx.load("resnet.onnx")  # load onnx model
onnx.checker.check_model(model_onnx)  # check onnx model

# Simplify
model_onnx, check = onnxsim.simplify(model_onnx)
assert check, 'assert check failed'

# convert_float_to_float16
model_onnx = float16.convert_float_to_float16(model_onnx)
onnx.save(model_onnx, "resnet_float16.onnx")