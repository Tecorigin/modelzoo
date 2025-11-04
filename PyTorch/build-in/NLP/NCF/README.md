# Neural-Collaborative-Filtering

## 1. 模型概述
Neural collaborative filtering(NCF), is a deep learning based framework for making recommendations. The key idea is to learn the user-item interaction using neural networks. Check the following paper for details about NCF.
- Paper:Neural Collaborative Filtering.
- Github Code:https://github.com/yihong-chen/neural-collaborative-filtering

## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集
#### 2.2.1 获取数据集
Dataset URL:http://grouplens.org/datasets/movielens/1m/, 这些文件包含约3900部电影的1000209个匿名评级,由2000年加入MovieLens的6040名MovieLens用户制作。

#### 2.2.2 处理数据集
Directory structure
```
.
├── data	# 存放数据
├── data.py	#prepare train/test dataset
├── util.py	#some handy functions for model training etc.
├── metrics.py #evaluation metrics including hit ratio(HR) and NDCG
├── gmf.py #generalized matrix factorization model
├── mlp.py #multi-layer perceptron model
├── neumf.py #fusion of gmf and mlp
├── readme.md	
├── engine.py #training engine
└── train.py #entry point for train a NCF model
```

### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境。
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```
2. 安装python依赖。
    ```
    pip install -r requirements.txt
    ```

### 2.4 启动训练

1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/build-in/NLP/NCF
    ```

2. 运行训练。该模型支持单机单核组。

    ```
    export TORCH_SDAA_AUTOLOAD=cuda_migrate  #自动迁移环境变量
    max_step=100 python train.py
    ```

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./loss.py)）



