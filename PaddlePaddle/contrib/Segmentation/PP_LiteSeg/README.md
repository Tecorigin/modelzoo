# PP_LiteSeg

## 1. 模型概述



## 2. 快速开始

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 数据集准备
#### 2.2.1 数据集准备

我们在本项目中使用了 cityscapes 数据集。根据[官方文档](./pre_data_cn.md)准备数据集。


#### 2.2.3 数据集目录结构

数据集目录结构参考如下所示:

```
cityscapes/
|-- gtFine
|   |-- test
|   |-- train
|   `-- val
`-- leftImg8bit
    |-- test
    |-- train
    `-- val
```

### 2.3 构建环境

参考[官方安装说明](./docs/install_cn.md)进行安装

1. 执行以下命令，启动虚拟环境。
``` bash
cd PaddlePaddle/contrib/Segmentation/PP_LiteSeg
conda activate paddleseg
```
2. 安装依赖
``` bash
cd PaddleSeg
pip install -r requirements.txt
pip install -v -e .
```

### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd PaddlePaddle/contrib/Segmentation/PP_LiteSeg
    ```

2. 运行训练。
    ```
    export SDAA_VISIBLE_DEVICES=0,1,2,3
    export PADDLE_XCCL_BACKEND=sdaa
    timeout 120m python -m  paddle.distributed.launch --devices=0,1,2,3 tools/train.py  \
        --config configs/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k.yml \
        --save_interval 50 \
        --save_dir output
    ```


### 2.5 训练结果

- 可视化命令
    ```
    cd ./script
    python plot_curve.py
    ```
| 加速卡数量 | 模型 | 混合精度 | Batch Size | epoch | train_loss |
| --- | --- | --- | --- | --- | --- |
| 2 | PPLiteSeg | 否 | 4 | 2 | 6.6480 |