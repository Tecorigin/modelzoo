import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2

class VOCDataset(Dataset):
    def __init__(self, root_dir, datashape="320", image_set='test'):
        """
        初始化VOC数据集
        :param root_dir: Pascal VOC2017 数据集的根目录路径
        :param image_set: 选择加载的图像集 ('train', 'val', 'test')
        :param transform: 图像转换操作（如resize、normalize等）
        """
        self.root_dir = root_dir
        self.image_set = image_set

        transform = [
            Resize(datashape),
            SubtractMeans([123, 117, 104]),
            ToTensor()
        ]       
        transform = Compose(transform)

        self.transform = transform

        self.image_dir = os.path.join(root_dir, "JPEGImages")
        self.anno_dir = os.path.join(root_dir, "Annotations")
        self.image_set_file = os.path.join(root_dir, "ImageSets", "Main", f"{image_set}.txt")

        # 读取所有图片文件的名字
        with open(self.image_set_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]
        # 定义类别
        self.class_names = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
                            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 
                            'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 
                            'sofa', 'train', 'tvmonitor']
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        self.class_to_ind = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.image_ids)
    
    def get_img_info(self, index):
        img_id = self.image_ids[index]
        annotation_file = os.path.join(self.root_dir, "Annotations", "%s.xml" % img_id)
        anno = ET.parse(annotation_file).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}
    
    def _get_annotation(self, image_id):
        annotation_file = os.path.join(self.root_dir, "Annotations", "%s.xml" % image_id)
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_dict[class_name])
            is_difficult_str = obj.find('difficult').text
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def get_annotation(self, index):
        image_id = self.image_ids[index]
        return image_id, self._get_annotation(image_id)
    
    def __getitem__(self, idx):
        """
        返回给定索引的图像及其标注
        :param idx: 索引
        :return: 图像及对应的标注（边界框和类别标签）
        """
        image_id = self.image_ids[idx]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        # boxes = boxes[is_difficult == 0]
        # labels = labels[is_difficult == 0]
        img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        anno_path = os.path.join(self.anno_dir, f"{image_id}.xml")

        # 加载图像
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        
        if self.transform is not None:
            image, boxes, labels = self.transform(image, boxes, labels)

        # 返回图像、边界框及类别
        target = {
            "boxes": boxes,
            "labels": labels
        }
        
        return image, target, idx

# 创建数据集和数据加载器
def collate_fn(batch):
    """
    用于处理不同大小的目标数据，并处理图像索引
    :param batch: 批数据
    :return: 分别返回批量图像、标注及索引
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    indices = [item[2] for item in batch]  # 新增：提取每个数据的索引

    images = torch.stack(images, dim=0)  # 将图像堆叠为一个batch

    return images, targets, indices


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels
    

class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels
    

class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size,
                                   self.size))
        return image, boxes, labels


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
            if boxes is not None:
                boxes, labels = remove_empty_boxes(boxes, labels)
        return img, boxes, labels


def remove_empty_boxes(boxes, labels):
    """Removes bounding boxes of W or H equal to 0 and its labels

    Args:
        boxes   (ndarray): NP Array with bounding boxes as lines
                           * BBOX[x1, y1, x2, y2]
        labels  (labels): Corresponding labels with boxes

    Returns:
        ndarray: Valid bounding boxes
        ndarray: Corresponding labels
    """
    del_boxes = []
    for idx, box in enumerate(boxes):
        if box[0] == box[2] or box[1] == box[3]:
            del_boxes.append(idx)

    return np.delete(boxes, del_boxes, 0), np.delete(labels, del_boxes)
