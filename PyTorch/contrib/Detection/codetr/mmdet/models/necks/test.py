import torch
import torch_sdaa
import torch.nn as nn
from mmcv.cnn import build_conv_layer

def apply_conv_on_input(input_tensor):
    # 设置随机种子以确保结果一致
    torch.manual_seed(42)
    conv = build_conv_layer(
        dict(type='Conv2d'),  # 默认使用普通的 Conv2d
        in_channels=256,
        out_channels=256,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True
    ).to('cpu')

    device = input_tensor.device

    # 如果有可用的 GPU，将卷积层移动到 GPU
    if torch.sdaa.is_available():
        conv = conv.to(device)
        input_tensor = input_tensor.to(device)

    # 应用卷积层于输入
    output = conv(input_tensor)
    return output

# 创建一个示例输入张量
inputs = [torch.ones(1, 256, 64, 64).to('sdaa:0')]  # 批次大小为 1，通道为 256，64x64 的输入

# 调用函数
output = apply_conv_on_input(inputs[0])

# 打印输出的形状
print(output)