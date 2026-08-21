import torch
import torch.nn as nn
import math

class SafeGroupConv2d(nn.Module):
    """
    修改版：在初始化时强制对齐原生卷积的随机权重。
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(SafeGroupConv2d, self).__init__()
        
        assert in_channels % groups == 0
        assert out_channels % groups == 0
        
        self.groups = groups
        self.convs = nn.ModuleList()
        
        in_channels_per_group = in_channels // groups
        out_channels_per_group = out_channels // groups

        # =======================================================
        # 【核心魔法】：影子层策略 (Shadow Layer Strategy)
        # 1. 创建一个临时的、原生的、带 groups 的卷积层
        #    它的唯一作用就是按照 PyTorch 标准逻辑消耗随机种子
        # =======================================================
        shadow_conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=groups,  # 这里用原始的 groups
            bias=bias
        )
        
        # 2. 获取这个影子层初始化好的权重 (Shape: [Out, In/G, K, K])
        #    因为此时全局 Seed 是固定的，所以这里生成的权重和 CUDA 上的一模一样
        master_weight = shadow_conv.weight.data
        
        # 3. 将权重在输出通道维度(dim=0)切分成 groups 份
        #    每一份 Shape: [Out/G, In/G, K, K]
        split_weights = torch.chunk(master_weight, groups, dim=0)
        
        if bias:
            master_bias = shadow_conv.bias.data
            split_bias = torch.chunk(master_bias, groups, dim=0)

        # =======================================================
        # 4. 创建实际运行的小卷积，并将权重“塞”进去
        # =======================================================
        for i in range(groups):
            # 创建小卷积 (groups=1, SDAA 支持)
            mini_conv = nn.Conv2d(
                in_channels_per_group, 
                out_channels_per_group, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding, 
                dilation=dilation, 
                groups=1, 
                bias=bias
            )
            
            # 【强制覆盖权重】
            mini_conv.weight.data = split_weights[i].clone()
            
            if bias:
                mini_conv.bias.data = split_bias[i].clone()
                
            self.convs.append(mini_conv)
            
        # 5. 影子层完成了它的使命，在此处会被自动回收，不占用训练显存

    def forward(self, x):
        x_splits = torch.chunk(x, self.groups, dim=1)
        results = []
        for i, conv in enumerate(self.convs):
            out = conv(x_splits[i])
            results.append(out)
        return torch.cat(results, dim=1)


class SKConv(nn.Module):
    def __init__(self, channels, branches=2, groups=32, reduce=16, stride=1, len=32):
        super(SKConv, self).__init__()
        len = max(channels // reduce, len)
        self.convs = nn.ModuleList([])

        for i in range(branches):
            # SKNet 的核心机制：不同分支使用不同的 dilation 和 padding
            dilation = 1 + i
            padding = 1 + i
            
            # === 修改逻辑开始 ===
            # 检测是否会触发 TecoDNN 的不支持配置 (groups > 1 且 dilation > 1)
            if groups > 1 and dilation > 1:
                # 使用自定义的拆解版卷积，避开报错
                conv_layer = SafeGroupConv2d(
                    channels, channels, kernel_size=3, stride=stride,
                    padding=padding, dilation=dilation, groups=groups, bias=False
                )
            else:
                # 正常情况（如 dilation=1）继续使用原生算子，保持最高性能
                conv_layer = nn.Conv2d(
                    channels, channels, kernel_size=3, stride=stride,
                    padding=padding, dilation=dilation, groups=groups, bias=False
                )
            # === 修改逻辑结束 ===

            self.convs.append(nn.Sequential(
                conv_layer,
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ))
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(channels, len, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(len),
            nn.ReLU(inplace=True)
        )
        self.fcs = nn.ModuleList([])
        for i in range(branches):
            self.fcs.append(
                nn.Conv2d(len, channels, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = [conv(x) for conv in self.convs]
        x = torch.stack(x, dim=1)
        attention = torch.sum(x, dim=1)
        attention = self.gap(attention)
        attention = self.fc(attention)
        attention = [fc(attention) for fc in self.fcs]
        attention = torch.stack(attention, dim=1)
        attention = self.softmax(attention)
        x = torch.sum(x * attention, dim=1)
        return x


class SKUnit(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, branches=2, group=32, reduce=16, stride=1, len=32):
        super(SKUnit, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = SKConv(mid_channels, branches=branches, groups=group, reduce=reduce, stride=stride, len=len)

        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        residual = self.shortcut(residual)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x += residual
        return self.relu(x)


class sknet(nn.Module):
    def __init__(self, num_classes, num_block_lists=[3, 4, 6, 3]):
        super(sknet, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.stage_1 = self._make_layer(64, 128, 256, nums_block=num_block_lists[0], stride=1)
        self.stage_2 = self._make_layer(256, 256, 512, nums_block=num_block_lists[1], stride=2)
        self.stage_3 = self._make_layer(512, 512, 1024, nums_block=num_block_lists[2], stride=2)
        self.stage_4 = self._make_layer(1024, 1024, 2048, nums_block=num_block_lists[3], stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, in_channels, mid_channels, out_channels, nums_block, stride=1):
        layers = [SKUnit(in_channels, mid_channels, out_channels, stride=stride)]
        for _ in range(1, nums_block):
            layers.append(SKUnit(out_channels, mid_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.basic_conv(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return x


def SKNet(num_classes=1000, depth=50):
    assert depth in [50, 101], 'depth invalid'
    key2blocks = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
    }
    model = sknet(num_classes, key2blocks[depth])
    return model

def Model(num_classes): # welo
    r"""Return your custom model
    """
    return SKNet(num_classes=num_classes)

    