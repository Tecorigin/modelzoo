import onnx
import onnxsim        # 用于简化模型
from onnxconverter_common import float16  # 用于将模型转为float16
import ssl
import torch
import torchvision
ssl._create_default_https_context = ssl._create_unverified_context
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='/mnt/nvme/common/user_data/Yxx/vgg16-397923af.pth', help='onnx path')

    opt = parser.parse_args()
    return opt

opt = parse_opt()
# 初始化并加载自定义 .pth 文件的 VGG16 模型
vgg16 = torchvision.models.vgg16(weights=False)  # 设置pretrained=False，避免预加载
checkpoint = torch.load(opt.ckpt)  # 将路径改为你的.pth文件路径
vgg16.load_state_dict(checkpoint)  # 加载.pth文件的权重
vgg16.eval()

# 初始化dummy_input
dummy_input = torch.randn(1, 3, 224, 224)

# 静态shape导出
torch.onnx.export(vgg16,
                  dummy_input,
                  "vgg16.onnx",
                  opset_version=12,         	# ONNX opset版本
                  input_names=['input'],    	# 输入名称
                  output_names=['output'],  	# 输出名称
                  do_constant_folding=True, 	# 是否执行常量折叠优化
                  dynamic_axes=None,	        # 是否使用动态shape，不使用默认为None
                 )

# # 动态shape导出（推荐）
dynamic_dims = {'input': {0: 'batch'},
                'output': {0: 'batch'}}

torch.onnx.export(vgg16,
                  dummy_input,
                  "vgg16_dyn.onnx",
                  opset_version=12,         	# ONNX opset版本
                  input_names=['input'],    	# 输入名称
                  output_names=['output'],  	# 输出名称
                  do_constant_folding=True, 	# 是否执行常量折叠优化
                  dynamic_axes=dynamic_dims,	# 使用动态shape
                 )

# 检查导出的模型
model_onnx = onnx.load("vgg16_dyn.onnx")  # 加载 ONNX 模型
onnx.checker.check_model(model_onnx)       # 检查模型

# 模型简化
model_onnx, check = onnxsim.simplify(model_onnx)
assert check, 'assert check failed'

# 转换模型为 float16
model_onnx = float16.convert_float_to_float16(model_onnx)
onnx.save(model_onnx, "vgg16_float16_dyn.onnx")  # 保存简化且转换后的模型

