# Big_Transfer_ResNetV2(BiT)

## 1. 模型概述
Big_Transfer_ResNetV2(BiT) 是一种通用视觉表示学习方法，来自 Google Research 的论文 *[Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370)*，由 Alexander Kolesnikov 等人在 2019 年发表。BiT 模型在 [ILSVRC-2012](http://www.image-net.org/challenges/LSVRC/2012/) 和 [ImageNet-21k](http://www.image-net.org/) 数据集上进行预训练，支持在多个深度学习框架（如 TensorFlow 2、PyTorch 和 Jax/Flax）上进行微调。原始代码仓库位于 [https://github.com/google-research/big_transfer](https://github.com/google-research/big_transfer)。

本项目适配了 BiT 模型，提供在 PyTorch 框架下的训练和微调支持，适用于分类任务等场景。有关知识蒸馏的更新模型（BiT-R50x1 蒸馏自 BiT-M-R152x2）可参考论文 *[Knowledge distillation: A good teacher is patient and consistent](https://arxiv.org/abs/2106.05237)*。

## 2. 快速开始
使用 BiT 模型执行训练的主要流程如下：
1. 基础环境安装：完成训练前的环境检查和安装。
2. 获取数据集：获取训练所需的数据集。
3. 构建环境：配置模型运行环境。
4. 启动训练：运行训练脚本。

### 2.1 基础环境安装
请参考基础环境安装章节，完成训练前的基础环境检查和安装. 

### 2.2 准备数据集
#### 2.2.1 获取数据集
BiT  使用 ImageNet 数据集，该数据集为开源数据集，可从 [ImageNet](https：//image-net.org/) 下载。

#### 2.2.2 处理数据集
具体配置方式可参考：https：//blog.csdn.net/xzxg001/article/details/142465729

### 2.3 构建环境

所使用的环境下已经包含 PyTorch 框架虚拟环境
1. 执行以下命令，启动虚拟环境. 
```
conda activate torch_env
```

2. 安装python依赖
```
pip install -r requirements.txt
```

### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录. 
```
cd <ModelZoo_path>/PyTorch/contrib/Classification/Big_Transfer_ResNetV2(BiT)/run_scripts
```
2. 运行训练. 该模型支持单机单卡。
```shell
python3 -m bit_pytorch.train \
--name imagenet_`date +%F_%H%M%S` \
--model BiT-M-R50x1 \
--logdir /tmp/bit_logs \
--dataset imagenet2012 \
--datadir /data/teco-data/imagenet \
--batch 4 \
--base_lr 0.00025 \
--workers 8 \
--eval_every 100 \
--batch_split 4 \
```
更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 
![训练loss曲线](./run_scripts/loss.jpg)

MeanRelativeError: -0.001270241089732237
MeanAbsoluteError: -0.012677725000000008
Rule,mean_absolute_error -0.012677725000000008
pass mean_relative_error=np.float64(-0.001270241089732237) <= 0.05 or mean_absolute_error=np.float64(-0.012677725000000008) <= 0.0002