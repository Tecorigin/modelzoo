# Res2Net50_14w_8s

## 1. 模型概述
Res2Net50_14w_8s 是一种深度卷积神经网络模型，特别设计用于提高计算机视觉任务中的特征表达能力。其核心创新在于引入了多尺度的残差网络结构，并且通过增加尺度数量来提高模型的表达能力。Res2Net 在保持计算效率的同时，能够更好地捕捉图像中的多尺度信息，具有较好的泛化能力。
源码链接:https://github.com/Res2Net/Res2Net-PretrainedModels/blob/master/res2net.py

## 2. 快速开始

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 数据集准备

Res2Net50_14w_8s运行在ImageNet数据集上，数据集配置可以参考https://blog.csdn.net/xzxg001/article/details/142465729

### 2.3 构建环境
所使用的环境下已经包含PyTorch框架虚拟环境
1. 执行以下命令，启动虚拟环境。
``` bash
cd <ModelZoo_path>/PyTorch/contrib/Classification/Res2Net50_14w_8s

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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/Res2Net50_14w_8s
    ```


- 单机单核组
    ```
  SDAA_LAUNCH_BLOCKING=1 python train.py --batch_size 128 --epochs 10 --distributed False --dataset_path /data/datasets/imagenet
  --lr 0.1 --autocast True
    ```
- 单机单卡
    ```
   python -m torch.distributed.launch --nproc_per_node=4 train.py --dataset_path /data/datasets/imagenet \
   --batch_size 128 --epochs 10 --distributed True --lr 0.1 --autocast True
   ```

### 2.5 训练结果



| 芯片 |卡  | 模型 |  混合精度 |Batch size|Shape| 
|:-:|:-:|:-:|:-:|:-:|:-:|
|SDAA|1| Res2Net50_14w_8s |是|128|224*224|
