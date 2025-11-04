import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

__all__ = ['DPN_Fast', 'dpn_fast']

def dpn_fast(num_classes=1000):
    return DPN_Fast(num_init_features=16, k_R=32, G=8, k_sec=(1, 2, 6, 1), inc_sec=(4, 8, 8, 32), num_classes=num_classes)

class DualPathBlock(nn.Module):
    def __init__(self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, G, _type='normal'):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c

        if _type == 'proj':
            key_stride = 1
            self.has_proj = True
        elif _type == 'down':
            key_stride = 2
            self.has_proj = True
        else:
            key_stride = 1
            self.has_proj = False

        if self.has_proj:
            self.c1x1_w = self.BN_ReLU_Conv(in_chs=in_chs, out_chs=num_1x1_c, kernel_size=1, stride=key_stride)

        self.layers = nn.Sequential(OrderedDict([
            ('c1x1_a', self.BN_ReLU_Conv(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)),
            ('c3x3_b', self.BN_ReLU_Conv(in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3, stride=key_stride, padding=1, groups=G)),
            ('c1x1_c', self.BN_ReLU_Conv(in_chs=num_3x3_b, out_chs=num_1x1_c, kernel_size=1, stride=1)),
        ]))

    def BN_ReLU_Conv(self, in_chs, out_chs, kernel_size, stride, padding=0, groups=1):
        return nn.Sequential(OrderedDict([
            ('norm', nn.BatchNorm2d(in_chs)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv', nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups, bias=False)),
        ]))

    def forward(self, x):
        if isinstance(x, list):
            x = x[0]  # 仅使用残差路径，丢弃密集路径
        if self.has_proj:
            data_o = self.c1x1_w(x)
        else:
            data_o = x

        out = self.layers(x)
        return data_o + out  # 仅保留残差连接

class DPN_Fast(nn.Module):
    def __init__(self, num_init_features=16, k_R=32, G=8, k_sec=(1, 2, 6, 1), inc_sec=(4, 8, 8, 32), num_classes=1000):
        super(DPN_Fast, self).__init__()
        blocks = OrderedDict()

        # conv1
        blocks['conv1'] = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),  # stride=1 保留空间维度
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
        )

        # conv2
        bw = 64
        inc = inc_sec[0]
        R = int((k_R * bw) / 256)
        blocks['conv2_1'] = DualPathBlock(num_init_features, R, R, bw, inc, G, 'proj')
        in_chs = bw
        for i in range(2, k_sec[0] + 1):
            blocks['conv2_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, G, 'normal')
            in_chs = bw

        self.features = nn.Sequential(blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应池化，确保输出为 1x1
        self.classifier = nn.Linear(in_chs, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).reshape(x.size(0), -1)  # 使用 reshape 替代 view
        x = self.classifier(x)
        return x