import torch
import torch.nn as nn
import math
import torch.nn.functional as F


# 定义自注意力模块
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        # 修改卷积层，确保输入通道数是in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 查询、键、值
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)

        # 计算注意力分数
        query = query.view(query.size(0), query.size(1), -1)
        key = key.view(key.size(0), key.size(1), -1)
        value = value.view(value.size(0), value.size(1), -1)

        # 点积注意力计算
        attention_scores = torch.bmm(query.permute(0, 2, 1), key)  # [batch_size, height*width, height*width]
        attention_scores = F.softmax(attention_scores, dim=-1)

        # 加权值
        out = torch.bmm(attention_scores, value.permute(0, 2, 1))
        out = out.view(x.size(0), self.in_channels, x.size(2), x.size(3))

        # 进行加权输出
        out = self.gamma * out + x
        return out


# 定义 DehazeNet 网络并添加自注意力机制
class dehaze_net(nn.Module):
    def __init__(self):
        super(dehaze_net, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
        self.e_conv2 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(6, 3, 5, 1, 2, bias=True)
        self.e_conv4 = nn.Conv2d(6, 3, 7, 1, 3, bias=True)
        # 将 e_conv5 的输出通道数改为 12
        self.e_conv5 = nn.Conv2d(12, 12, 3, 1, 1, bias=True)  # 改为 12 输出通道

        # 添加自注意力模块
        self.attention = SelfAttention(12)  # 使用 12 通道的输入（从 concat3）

    def forward(self, x):
        source = []
        source.append(x)

        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))

        concat1 = torch.cat((x1, x2), 1)
        x3 = self.relu(self.e_conv3(concat1))

        concat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.e_conv4(concat2))

        concat3 = torch.cat((x1, x2, x3, x4), 1)
        x5 = self.relu(self.e_conv5(concat3))

        print(f"x5 shape before attention: {x5.shape}")  # 打印 x5 的形状，确保它是 (batch_size, 12, height, width)

        # 在特征融合后，添加自注意力机制
        x5 = self.attention(x5)

        # 最终图像生成
        clean_image = self.relu((x5 * x) - x5 + 1)

        return clean_image


