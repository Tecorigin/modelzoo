import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset
from PIL import Image

# 自定义 Dataset 类
class CustomImageNetDataset(Dataset):
    def __init__(self, root_dir, selected_classes, transform=None):
        self.root_dir = root_dir
        self.selected_classes = selected_classes
        self.transform = transform

        self.image_paths = []
        self.labels = []
        for idx, class_name in enumerate(self.selected_classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if img_path.endswith(('.JPEG', '.jpg', '.jpeg')):
                    self.image_paths.append(img_path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# LeNet 网络
class LeNet(nn.Module):
    def __init__(self, num_classes=20):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)

        # 计算 Flatten 层的大小
        self._to_linear = None
        self._get_flatten_size()  # 自动计算 Flatten 层大小

        self.fc1 = nn.Linear(self._to_linear, 120)
        self.bn_fc1 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn_fc2 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, num_classes)  # 动态支持不同类别数

    def _get_flatten_size(self):
        dummy_input = torch.randn(1, 3, 224, 224)  # 224x224 输入
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(dummy_input))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        self._to_linear = x.numel()  # 计算 Flatten 层的大小

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = x.reshape(x.shape[0], -1)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)

        return x
