# TinyVit 

## 1. 模型概述
TinyViT 是一种轻量化的视觉Transformer模型，通过结合卷积神经网络（CNN）的局部建模优势与Transformer的全局建模能力，实现了在保持较低计算成本和参数量的同时，取得优异的图像识别性能。TinyViT 结构紧凑，适用于移动端和边缘设备，在图像分类、目标检测等多种视觉任务中表现出良好的效果。
源码链接:https://github.com/wkcn/tinyvit

## 2. 快速开始

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 数据集准备

TinyVit运行在ImageNet数据集上，数据集配置可以参考https://blog.csdn.net/xzxg001/article/details/142465729

### 2.3 构建环境
所使用的环境下已经包含PyTorch框架虚拟环境
1. 执行以下命令，启动虚拟环境。
``` bash
cd <ModelZoo_path>/PyTorch/contrib/Classification/TinyVit  

conda activate torch_env

# 执行以下命令验证环境是否正确，正确则会打印如下版本信息
python -c "import torch_sdaa"
```

2. 安装python依赖
``` 
pip install -r requirements.txt
```
### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Classification/TinyVit 
    ```


- 单机单核组
    ```
   python train.py --batch_size 128 --epochs 6 --distributed False --dataset_path /data/datasets/imagenet
  --lr 0.0005 --autocast True 
    ```
- 单机单卡
    ```
  python -m torch.distributed.launch --nproc_per_node=4 train.py --dataset_path /data/datasets/imagenet \
  --batch_size 128 --epochs 6 --distributed True --lr 0.0005 --autocast True
   ```

### 2.5 训练结果



| 芯片 |卡  | 模型 |  混合精度 |Batch size|Shape| 
|:-:|:-:|:-:|:-:|:-:|:-:|
|SDAA|1| TinyVit   |是|128|224*224|
