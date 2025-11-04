# Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose

## 模型简介：
本模型是一个基于 PyTorch 实现的轻量级人体姿态估计模型仓库。以下是该仓库的简要介绍


## 数据集准备和模型权重：
本仓库使用的数据集为COCO 2017数据集，该数据集包含了2017年的图像和标注信息。
下载COCO 2017数据集：[URL_ADDRESS1. 下载COCO 2017数据集：[http://cocodataset.org/#download](http://cocodataset.org/#download) (train, val, annotations) 并解压到`<COCO_HOME>`文件夹中。将COCO 2017数据集的图像和标注信息分别存储到`data/coco/images`和`data/coco/annotations`文件夹中。
从https://github.com/marvis/pytorch-mobilenet (sgd option)下载预训练权重mobilenet_sgd_68.848.pth.tar，如果下载失败，可以从https://drive.google.com/file/d/18Ya27IAhILvBHqV_tDp0QjDFvsNNy-hv/view?usp=sharing下载。
将预训练权重mobilenet_sgd_68.848.pth.tar保存到`weights`文件夹中。

## 安装依赖：
`pip install -r requirements.txt`

## 模型训练：
python scripts/prepare_train_labels.py --labels <COCO_HOME>/annotations/person_keypoints_train2017.json
python scripts/make_val_subset.py --labels <COCO_HOME>/annotations/person_keypoints_val2017.json
ython train.py --train-images-folder <COCO_HOME>/train2017/ --prepared-train-labels prepared_train_annotation.pkl --val-labels val_subset.json --val-images-folder <COCO_HOME>/val2017/ --checkpoint-path <path_to>/mobilenet_sgd_68.848.pth.tar --from-mobilenet