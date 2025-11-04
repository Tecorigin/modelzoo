import ssl
import torch
import onnx
import onnxsim        # 用于简化模型
from onnxconverter_common import float16  # 用于将模型转为float16

# 禁用SSL验证
ssl._create_default_https_context = ssl._create_unverified_context

# 安装并导入预训练模型库
import pretrainedmodels

# 加载InceptionV4模型
inceptionv4 = pretrainedmodels.__dict__['inceptionv4'](num_classes=1000, pretrained='imagenet')
inceptionv4.eval()

# 初始化dummy_input
dummy_input = torch.randn(1, 3, 299, 299)  # InceptionV4通常使用299x299输入尺寸

# 动态shape导出
dynamic_dims = {'input': {0: 'batch', 2: 'height', 3: 'width'},
                'output': {0: 'batch'}}

torch.onnx.export(inceptionv4,
                  dummy_input,
                  "inceptionv4_dyn.onnx",
                  opset_version=12,         	# ONNX opset版本
                  input_names=['input'],    	# 输入名称
                  output_names=['output'],  	# 输出名称
                  do_constant_folding=True, 	# 是否执行常量折叠优化
                  dynamic_axes=dynamic_dims,	# 使用动态shape
                 )

# 检查导出的模型
model_onnx = onnx.load("inceptionv4_dyn.onnx")  # 加载 ONNX 模型
onnx.checker.check_model(model_onnx)       # 检查模型

# 模型简化
model_onnx, check = onnxsim.simplify(model_onnx)
assert check, 'assert check failed'

# 转换模型为 float16
model_onnx = float16.convert_float_to_float16(model_onnx)
onnx.save(model_onnx, "inceptionv4_float16_dyn.onnx")  # 保存简化且转换后的模型