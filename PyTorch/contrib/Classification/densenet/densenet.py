import torch
import torch.nn as nn
import math

class DenseConnectLayerStandard(nn.Module):
    def __init__(self, nChannels, opt):
        super(DenseConnectLayerStandard, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(nChannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(nChannels, opt.growthRate, kernel_size=3, stride=1, padding=1, bias=False)
        )
        if opt.dropRate > 0:
            self.layer.add_module('dropout', nn.Dropout2d(opt.dropRate))
        self.nChannels = nChannels
        self.growthRate = opt.growthRate

    def forward(self, x):
        out = self.layer(x)
        return torch.cat([x, out], dim=1)  # 拼接输入和输出，增加 growthRate 个通道

class DenseConnectLayerCustom(nn.Module):
    def __init__(self, nChannels, opt):
        super(DenseConnectLayerCustom, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(nChannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(nChannels, opt.growthRate, kernel_size=3, stride=1, padding=1, bias=False)
        )
        if opt.dropRate > 0:
            self.layer.add_module('dropout', nn.Dropout2d(opt.dropRate))
        self.nChannels = nChannels
        self.growthRate = opt.growthRate

    def forward(self, x):
        out = self.layer(x)
        return torch.cat([x, out], dim=1)  # 显式拼接输入和输出

def add_layer(model, nChannels, opt):
    if opt.optMemory >= 3:
        model.add_module(f'dense_layer_{len(model)}', DenseConnectLayerCustom(nChannels, opt))
    else:
        model.add_module(f'dense_layer_{len(model)}', DenseConnectLayerStandard(nChannels, opt))

def add_transition(model, nChannels, nOutChannels, opt, last=False, pool_size=8):
    layers = []
    layers.append(nn.BatchNorm2d(nChannels))
    layers.append(nn.ReLU(inplace=True))
    
    if last:
        layers.append(nn.AvgPool2d(kernel_size=pool_size))
        layers.append(nn.Flatten())
    else:
        layers.append(nn.Conv2d(nChannels, nOutChannels, kernel_size=1, stride=1, padding=0, bias=False))
        if opt.dropRate > 0:
            layers.append(nn.Dropout2d(opt.dropRate))
        layers.append(nn.AvgPool2d(kernel_size=2))

    transition = nn.Sequential(*layers)
    model.add_module(f'transition_{len(model)}', transition)

def add_dense_block(model, nChannels, opt, N):
    for i in range(int(N)):
        add_layer(model, nChannels, opt)
        nChannels += opt.growthRate  # 每次 Dense 层增加 growthRate 个通道
    return nChannels

class DenseNet(nn.Module):
    def __init__(self, opt):
        super(DenseNet, self).__init__()
        growthRate = opt.growthRate
        dropRate = opt.dropRate
        reduction = opt.reduction
        bottleneck = opt.bottleneck
        N = (opt.depth - 4) / 3
        if bottleneck:
            N = N / 2

        nChannels = 2 * growthRate  # 初始通道数
        self.model = nn.Sequential()

        if opt.dataset in ['cifar10', 'cifar100']:
            self.model.add_module('conv0', nn.Conv2d(3, nChannels, kernel_size=3, stride=1, padding=1, bias=False))

            nChannels = add_dense_block(self.model, nChannels, opt, N)
            add_transition(self.model, nChannels, int(math.floor(nChannels * reduction)), opt)
            nChannels = int(math.floor(nChannels * reduction))

            nChannels = add_dense_block(self.model, nChannels, opt, N)
            add_transition(self.model, nChannels, int(math.floor(nChannels * reduction)), opt)
            nChannels = int(math.floor(nChannels * reduction))

            nChannels = add_dense_block(self.model, nChannels, opt, N)
            add_transition(self.model, nChannels, nChannels, opt, last=True, pool_size=8)

        elif opt.dataset == 'imagenet':
            if opt.depth == 121:
                stages = [6, 12, 24, 16]
            elif opt.depth == 169:
                stages = [6, 12, 32, 32]
            elif opt.depth == 201:
                stages = [6, 12, 48, 32]
            elif opt.depth == 161:
                stages = [6, 12, 36, 24]
            else:
                stages = [opt.d1, opt.d2, opt.d3, opt.d4]

            self.model.add_module('conv0', nn.Conv2d(3, nChannels, kernel_size=7, stride=2, padding=3, bias=False))
            self.model.add_module('bn0', nn.BatchNorm2d(nChannels))
            self.model.add_module('relu0', nn.ReLU(inplace=True))
            self.model.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

            nChannels = add_dense_block(self.model, nChannels, opt, stages[0])
            add_transition(self.model, nChannels, int(math.floor(nChannels * reduction)), opt)
            nChannels = int(math.floor(nChannels * reduction))

            nChannels = add_dense_block(self.model, nChannels, opt, stages[1])
            add_transition(self.model, nChannels, int(math.floor(nChannels * reduction)), opt)
            nChannels = int(math.floor(nChannels * reduction))

            nChannels = add_dense_block(self.model, nChannels, opt, stages[2])
            add_transition(self.model, nChannels, int(math.floor(nChannels * reduction)), opt)
            nChannels = int(math.floor(nChannels * reduction))

            nChannels = add_dense_block(self.model, nChannels, opt, stages[3])
            add_transition(self.model, nChannels, nChannels, opt, last=True, pool_size=7)

        if opt.dataset == 'cifar10':
            self.model.add_module('fc', nn.Linear(nChannels, 10))
        elif opt.dataset == 'cifar100':
            self.model.add_module('fc', nn.Linear(nChannels, 100))
        elif opt.dataset == 'imagenet':
            self.model.add_module('fc', nn.Linear(nChannels, 1000))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)

def create_model(opt):
    model = DenseNet(opt)
    if opt.cudnn == 'deterministic':
        for m in model.modules():
            if hasattr(m, 'set_mode'):
                m.set_mode(1, 1, 1)
    return model 