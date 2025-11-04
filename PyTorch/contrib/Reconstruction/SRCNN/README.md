
# 超分辨率重建任务SRCNN在Pytorch当中的实现
## 1. 模型概述
SRCNN 是将低分辨率图像至高分辨率图像的重建模型，来自于https://github.com/yjn780/SRCNN-pytorch。


## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 准备数据集
#### 2.2.1 数据集介绍
本次超分辨重建任务的数据集的类型是高分辨图像和低分辨图像的数据对

91-image数据集，包含91对.bmp格式的图像

Set5-image数据集，包含5对.bmp格式的图像
#### 2.2.2 数据集下载
数据集中训练集为91-image_x3,测试集为Set5_x3
训练集下载链接: https://www.dropbox.com/s/curldmdf11iqakd/91-image_x3.h5?dl=0

测试集下载链接:https://www.dropbox.com/s/58ywjac4te3kbqq/Set5_x3.h5?dl=0

#### 2.2.3 数据集存放地址
训练数据存放地址为 BLAH_BLAH/91-image_x3.h5

测试数据存放地址为 BLAH_BLAH/Set5_x3_x3.h5


### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境
1. 执行以下命令，启动虚拟环境。
    ```       
    conda activate torch_env
    ```

2. 安装python依赖
    ```
    pip install -r requirements.txt
    ```
### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Reconstruction/SRCNN
    ```

2. 运行训练。该模型支持单机单卡.

   单机单SPA
   ```
   python run_scripts/run_train.py --nproc_per_node 1 --train_file "BLAH_BLAH/91-image_x3.h5" --eval_file "BLAH_BLAH/Set5_x3.h5" --outputs_dir "BLAH_BLAH/outputs" --scale 3 --lr 1e-4 --batch_size 512 --num_epochs 50 --num_workers 8 --seed 123 --use_amp True --use_ddp False
   ```

    单机单卡（DDP）
   ```
   python -m torch.distributed.launch run_scripts/run_train.py --nproc_per_node 3 --train_file "BLAH_BLAH/91-image_x3.h5" --eval_file "BLAH_BLAH/Set5_x3.h5" --outputs_dir "BLAH_BLAH/outputs" --scale 3 --lr 1e-4 --batch_size 512 --num_epochs 50 --num_workers 1 --seed 123 --use_amp True --use_ddp True
   ```

   更多训练参数参考[README](run_scripts/README.md)

### 2.5 训练结果

训练条件

| 芯片   | 卡 | 模型          | 混合精度 | batch size |
|------- |----|--------------|----------|------------|
| SDAA   | 1  | SRCNN | 是    | 512        | 

训练结果量化指标如下表所示

|     | epoch | 混合精度 | batch size | 最佳PSNR 峰值信噪比|
|-------|-------|------|------------|---------|
| 单机单SPA     | 50    | 是    | 512        | 32.28  |


|   | epoch | 混合精度 | batch size | 最佳PSNR 峰值信噪比|
|-------|-------|------|------------|---------|
| 单机单卡（DDP）    | 50    | 是    | 512        | 32.96  |

