import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self,
                 root_dir: str,
                 txt_name: str,
                 transform=None):
        self.root_dir = root_dir
        self.txt_name = txt_name
        self.transform = transform
        
        images_dir = os.path.join(root_dir, "images")
        assert os.path.exists(images_dir), "dir:'{}' not found.".format(images_dir)

        txt_path = os.path.join(root_dir, txt_name)
        assert os.path.exists(txt_path), "file:'{}' not found.".format(txt_path)

        # 读取txt文件内容
        with open(txt_path, "r") as f:
            lines = f.readlines()

        self.img_paths = []
        self.img_labels = []
        for line in lines:
            file_name, label = line.strip().split()
            self.img_paths.append(os.path.join(images_dir, file_name))
            self.img_labels.append(int(label))

        self.total_num = len(self.img_paths)
        self.labels = set(self.img_labels)

        self.transform = transform

    def __len__(self):
        return self.total_num

    def __getitem__(self, item):
        img = Image.open(self.img_paths[item])
        if img.mode == 'L':
            img = img.convert('RGB')
        label = self.img_labels[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
