# Pytorch Implementation of PointNet and PointNet++ 

## 模型简介
PointNet是2016年提出的一种基于点云的深度学习模型，其主要思想是将点云数据作为输入，通过学习点云的局部特征和全局特征，从而实现对点云的分类、分割和重建等任务。

## 数据准备
下载对齐的**ModelNet**数据集[这里]URL_ADDRESS下载对齐的**ModelNet**数据集[这里](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip)，并将其保存到`data/modelnet40_normal_resampled/`目录下。
## 开始训练
您可以使用以下代码运行不同的模式。
* 如果您想使用离线处理数据，您可以在第一次运行时使用`--process_data`。您可以从[这里](URL_ADDRESS)下载预处理的数据，并将其保存到`data/modelnet40_normal_resampled/`目录下。
* 如果您想在ModelNet10上训练，您可以使用`--num_category 10`。
```shell
# ModelNet40
## 选择不同的模型在./models
## e.g., pointnet2_ssg without normal features
python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg
python test_classification.py --log_dir pointnet2_cls_ssg
## e.g., pointnet2_ssg with normal features
python train_classification.py --model pointnet2_cls_ssg --use_normals --log_dir pointnet2_cls_ssg_normal
python test_classification.py --use_normals --log_dir pointnet2_cls_ssg_normal
## e.g., pointnet2_ssg with uniform sampling
python train_classification.py --model pointnet2_cls_ssg --use_uniform_sample --log_dir pointnet2_cls_ssg_uniform
python test_classification.py --use_uniform_sample --log_dir pointnet2_cls_ssg_uniform
```
