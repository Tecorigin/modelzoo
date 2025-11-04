# ST-GCN

## 1. 模型概述

ST-GCN 即 Spatial Temporal Graph Convolutional Networks，是一种专门用于处理时空数据的图卷积神经网络模型，主要应用于人体动作识别等领域。它创新性地将图卷积网络（GCN）扩展到时空维度，通过构建空间图来表示人体关节点之间的关系，同时利用时间维度上的卷积操作捕捉动作的动态变化。ST-GCN 能够有效提取人体动作在空间和时间上的特征信息，在动作序列中建模关节点的空间依赖关系和时间演化模式。在人体动作识别、行为分析等计算机视觉任务中，ST-GCN 凭借其独特的时空建模能力，能够准确地对各种动作进行分类和理解，相比传统方法在处理动态时空数据时具有明显优势，已成为动作分析领域的重要基础模型之一，为智能视频分析、人机交互等应用提供了有力支持。

## 2. 快速开始

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 数据集准备
#### 2.2.1 数据集准备

我们在本项目中使用了 Kinetics-skeleton 数据集。链接：https://deepmind.com/research/open-source/open-source-datasets/kinetics/

#### 2.2.2 处理数据集
- 无需额外处理
- 数据集位置：`/data/datasets/20241122/data/Kinetics/kinetics-skeleton/`


#### 2.2.3 数据集目录结构

数据集目录结构参考如下所示:

```
kinetics-skeleton
    |-- train_data.npy
    |-- train_label.pkl
    |-- val_data.npy
    `-- val_label.pkl
```


### 2.3 构建环境
1. 执行以下命令，启动环境。

```sh
conda activate stgcn
```

2. 安装依赖
```
pip install requirements.txt
cd torchlight
python setup.py install
```

### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd PyTorch/contrib/Detection/st-gcn
    ```

2. 运行训练。
    ```
    export SDAA_VISIBLE_DEVICES=0,1,2,3
    python3 -m torch.distributed.launch --use-env --nproc_per_node=4 main.py recognition -c config/st_gcn/kinetics-skeleton/train.yaml | tee ./scripts/train_sdaa_3rd.log
    ```

### 2.5 训练结果

- 可视化命令
    ```
    cd ./scripts
    python plot_curve.py
    ```
 训练2h，loss震荡

| 加速卡数量 | 模型 | 混合精度 | Batch Size | epoch | train_loss |
| --- | --- | --- | --- | --- | --- |
| 2 | ST-GCN | 是 | 32 | 140 | - |