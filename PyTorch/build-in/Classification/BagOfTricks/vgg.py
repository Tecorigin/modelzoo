"""[1] Simonyan, Karen, and Andrew Zisserman. “Very Deep Convolutional 
       Networks for Large-Scale Image Recognition.” International Conference
       on Learning Representations, 2015."""


import torch
import torch.nn as nn

class BasicConv(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self._blocks = []
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class VGG(nn.Module):

    def __init__(self, blocks, num_class=100):
        super().__init__()
        self.input_channels = 3
        self.conv1 = self._make_layers(64, blocks[0])
        self.conv2 = self._make_layers(128, blocks[1])
        self.conv3 = self._make_layers(256, blocks[2])
        self.conv4 = self._make_layers(512, blocks[3])
        self.conv5 = self._make_layers(512, blocks[4])
    
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = x.view(x.size(0), -1)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)

        return x

    def _make_layers(self, output_channels, layer_num):
        layers = []
        while layer_num:
            layers.append(
                BasicConv(
                    self.input_channels, 
                    output_channels, 
                    kernel_size=3, 
                    padding=1, 
                    bias=False
                )
            )
            self.input_channels = output_channels
            layer_num -= 1
        layers.append(nn.MaxPool2d(2, stride=2))

        return nn.Sequential(*layers)

def vgg11(num_classes):
    return VGG([1, 1, 2, 2, 2], num_class=num_classes)

def vgg13(num_classes):
    return VGG([2, 2, 2, 2, 2], num_class=num_classes)

def vgg16(num_classes):
    return VGG([2, 2, 3, 3, 3], num_class=num_classes)

def vgg19(num_classes):
    return VGG([2, 2, 4, 4, 4], num_class=num_classes)

# welo: minimal VGG factory — clean, simple
def Model(num_classes=100, model_type='vgg16'):
    model_map = {
        'vgg11': vgg11,
        'vgg13': vgg13,
        'vgg16': vgg16,
        'vgg19': vgg19,
    }

    if model_type not in model_map:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model_map[model_type](num_classes=num_classes)