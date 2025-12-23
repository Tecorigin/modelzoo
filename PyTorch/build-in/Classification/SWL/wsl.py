# resnext_wsl_single_file.py
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

# ============================
# 1. WSL 预训练权重 URL
# ============================
model_urls = {
    'resnext101_32x8d':  'https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth',
    'resnext101_32x16d': 'https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth',
    'resnext101_32x32d': 'https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth',
    'resnext101_32x48d': 'https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth',
}

# ============================
# 2. 基础工具
# ============================
def conv3x3(in_planes, out_planes, stride=1, groups=1):
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=3, stride=stride,
        padding=1, groups=groups, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=1, stride=stride, bias=False
    )

# ============================
# 3. Bottleneck（ResNeXt 核心）
# ============================
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        width_per_group=64
    ):
        super().__init__()

        width = int(planes * (width_per_group / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1   = nn.BatchNorm2d(width)

        self.conv2 = conv3x3(
            width, width,
            stride=stride,
            groups=groups
        )
        self.bn2   = nn.BatchNorm2d(width)

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

# ============================
# 4. ResNet / ResNeXt 主体
# ============================
class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        groups=1,
        width_per_group=64
    ):
        super().__init__()

        self.inplanes = 64
        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1
        )

        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                width_per_group=self.width_per_group,
            )
        )
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes,
                    groups=self.groups,
                    width_per_group=self.width_per_group,
                )
            )

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ============================
# 5. 构造 ResNeXt-101 WSL
# ============================
def _resnext(arch, groups, width_per_group, progress=True, **kwargs):
    model = ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        groups=groups,
        width_per_group=width_per_group,
        **kwargs
    )
    state_dict = load_state_dict_from_url(
        model_urls[arch],
        progress=progress
    )
    model.load_state_dict(state_dict)
    return model

def resnext101_32x8d(num_classes=1000):
    return ResNet(
        Bottleneck,
        layers=[3, 4, 23, 3],
        num_classes=num_classes,
        groups=32,
        width_per_group=8,
    )


def resnext101_32x16d(num_classes=1000):
    return ResNet(
        Bottleneck,
        layers=[3, 4, 23, 3],
        num_classes=num_classes,
        groups=32,
        width_per_group=16,
    )


def resnext101_32x32d(num_classes=1000):
    return ResNet(
        Bottleneck,
        layers=[3, 4, 23, 3],
        num_classes=num_classes,
        groups=32,
        width_per_group=32,
    )


def resnext101_32x48d(num_classes=1000):
    return ResNet(
        Bottleneck,
        layers=[3, 4, 23, 3],
        num_classes=num_classes,
        groups=32,
        width_per_group=48,
    )


def Model(num_classes=1000, variant="32x8d"):
    """
    variant:
        - "32x8d"
        - "32x16d"
        - "32x32d"
        - "32x48d"
    """
    if variant == "32x8d":
        return resnext101_32x8d(num_classes=num_classes)
    elif variant == "32x16d":
        return resnext101_32x16d(num_classes=num_classes)
    elif variant == "32x32d":
        return resnext101_32x32d(num_classes=num_classes)
    elif variant == "32x48d":
        return resnext101_32x48d(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown variant: {variant}")
