# 3DUNet
## 1. 项目介绍

3DUNet是一种基于编码-解码结构的三维语义分割网络，广泛应用于医学图像分割等三维体积数据处理任务。它通过多层下采样和上采样操作，结合跳跃连接来保留空间细节信息，从而实现高精度的三维分割。3DUNet在医学图像分析中表现优异，例如在肿瘤分割、细胞边界预测等任务中展现了强大的性能。其网络结构支持多种损失函数和评估指标，适用于不同类型的三维分割任务。


## 2. 快速开始

### 2.1 基础环境安装
安装python依赖
``` bash
cd 3DUNet
# 克隆基础torch环境
conda create -n 3dunet --clone torch_env
# 链接源代码
pip install -e .
# install requirements
pip install -r requirements.txt
```

### 2.2 数据集准备


The data used for training can be downloaded from the following OSF project:
* training set: https://osf.io/9x3g2/
* validation set: https://osf.io/vs6gb/
* test set: https://osf.io/tn4xj/

### 2.3 启动训练
1. 在构建好的环境中，进入项目根目录。
    ```
   cd 3DUNet
    ```

- 单机单SPA训练
    ```
   train3dunet --config resources/3DUnet_confocal_boundary/train_config.yml
    ```
- 单机单卡训练（DDP）
    ```
   SDAA_VISIBLE_DEVICES=0,1,2,3 
   train3dunet --config resources/3DUnet_confocal_boundary/train_config.yml
    ```


### 2.5 训练结果


|加速卡数量  |模型 | 混合精度 |Batch size|Shape| AccTop1|
|:-:|:-:|:-:|:-:|:-:|:-:|
|1| 3DUNet|是|1|80 *170 *170| / |