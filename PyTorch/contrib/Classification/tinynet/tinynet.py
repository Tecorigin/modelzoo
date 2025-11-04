import torch
import torch.nn as nn
import math
import re

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEModule(nn.Module):
    def __init__(self, channels, reduction=0.25):
        super(SEModule, self).__init__()
        reduced_chs = max(1, int(channels * reduction))
        self.fc1 = nn.Conv2d(channels, reduced_chs, kernel_size=1, bias=True)
        self.act = Swish()
        self.fc2 = nn.Conv2d(reduced_chs, channels, kernel_size=1, bias=True)

    def forward(self, x):
        se = x.mean((2, 3), keepdim=True)
        se = self.fc1(se)
        se = self.act(se)
        se = self.fc2(se)
        se = torch.sigmoid(se)
        return x * se

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.has_se = se_ratio > 0
        expanded_channels = int(in_channels * expand_ratio)

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, expanded_channels, 1, bias=False))
            layers.append(nn.BatchNorm2d(expanded_channels))
            layers.append(Swish())
        
        layers.append(nn.Conv2d(expanded_channels, expanded_channels, kernel_size, 
                               stride=stride, padding=kernel_size//2, groups=expanded_channels, bias=False))
        layers.append(nn.BatchNorm2d(expanded_channels))
        layers.append(Swish())

        if self.has_se:
            layers.append(SEModule(expanded_channels, se_ratio))

        layers.append(nn.Conv2d(expanded_channels, out_channels, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)
        self.has_residual = (in_channels == out_channels and stride == 1)

    def forward(self, x):
        out = self.block(x)
        if self.has_residual:
            out = out + x
        return out

def decode_arch_def(arch_def, depth_multiplier=1.0, depth_trunc='round'):
    block_args = []
    for block_strings in arch_def:
        for block_str in block_strings:
            match = re.match(r'(\w+)_r(\d+)_k(\d+)_s(\d+)_e([\d.]+)_c(\d+)_se([\d.]+)', block_str)
            if not match:
                raise ValueError(f"Invalid block string: {block_str}")
            block_type, repeat, kernel_size, stride, expand_ratio, channels, se_ratio = match.groups()
            repeat = int(repeat)
            kernel_size = int(kernel_size)
            stride = int(stride)
            expand_ratio = float(expand_ratio)
            channels = int(channels)
            se_ratio = float(se_ratio)
            
            repeat = max(1, int(round(repeat * depth_multiplier)) if depth_trunc == 'round' else math.ceil(repeat * depth_multiplier))
            
            for _ in range(repeat):
                block_args.append({
                    'block_type': block_type,
                    'kernel_size': kernel_size,
                    'stride': stride,
                    'expand_ratio': expand_ratio,
                    'channels': channels,
                    'se_ratio': se_ratio,
                })
                stride = 1
    return block_args

def round_channels(channels, multiplier=1.0, divisor=8, min_channels=None):
    channels = int(channels * multiplier)
    if divisor:
        channels = max(divisor, (channels + divisor // 2) // divisor * divisor)
    if min_channels:
        channels = max(min_channels, channels)
    return channels

class TinyNet(nn.Module):
    def __init__(self, block_args, num_features, stem_size, channel_multiplier=1.0, input_size=(3, 224, 224), num_classes=1000):
        super(TinyNet, self).__init__()
        self.input_size = input_size

        self.stem = nn.Sequential(
            nn.Conv2d(input_size[0], stem_size, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_size),
            Swish()
        )

        layers = []
        in_channels = stem_size
        for args in block_args:
            out_channels = round_channels(args['channels'], channel_multiplier, 8, None)
            if args['block_type'] == 'ds':
                layers.append(nn.Conv2d(in_channels, out_channels, args['kernel_size'], 
                                       stride=args['stride'], padding=args['kernel_size']//2, 
                                       groups=in_channels, bias=False))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(Swish())
                if args['se_ratio'] > 0:
                    layers.append(SEModule(out_channels, args['se_ratio']))
            elif args['block_type'] == 'ir':
                layers.append(MBConvBlock(in_channels, out_channels, args['kernel_size'], 
                                         args['stride'], args['expand_ratio'], args['se_ratio']))
            in_channels = out_channels

        self.blocks = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

def _gen_tinynet(variant_cfg, channel_multiplier=1.0, depth_multiplier=1.0, depth_trunc='round', **kwargs):
    arch_def = [
        ['ds_r1_k3_s1_e1_c32_se0.25'],
        ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'], 
        ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'], 
        ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]
    block_args = decode_arch_def(arch_def, depth_multiplier, depth_trunc=depth_trunc)
    model = TinyNet(
        block_args=block_args,
        num_features=max(1280, round_channels(1280, channel_multiplier, 8, None)),
        stem_size=32,
        channel_multiplier=channel_multiplier,
        input_size=variant_cfg['input_size'],
        num_classes=variant_cfg.get('num_classes', 1000),
        **kwargs
    )
    return model

def tinynet(r=1.0, w=1.0, d=1.0, **kwargs):
    hw = int(224 * r)
    variant_cfg = {'input_size': (3, hw, hw), 'num_classes': 1000}
    model = _gen_tinynet(
        variant_cfg, channel_multiplier=w, depth_multiplier=d, **kwargs)
    return model