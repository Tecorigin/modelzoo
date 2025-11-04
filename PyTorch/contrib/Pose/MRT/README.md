# MRT Training

## 1. 模型概述
[MRT: Multi-Person 3D Motion Prediction with Multi-Range Transformers (NeurIPS 2021)](https://github.com/jiashunwang/MRT)提出了一种新颖的多人3D运动轨迹预测算法。算法引入了一个多范围Transformer模型，该模型包含一个用于个体运动的局部范围encoder和一个用于多人互动的全局范围encoder，最后通过Transformer decoder对全局信息进行解码和姿态轨迹预测。
<!-- toc -->

## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 构建Docker环境：介绍如何使用Dockerfile创建模型训练时所需的Docker环境。
3. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考[基础环境安装](../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。

### 2.2 构建Docker环境

1. 执行以下命令，进入Dockerfile所在目录。
    ```
    cd <modelzoo-dir>/PyTorch/contrib/Pose/MRT
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录

2. 执行以下命令，构建名为`mrt_t`的镜像。
   ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t  mrt_t
   ```

3. 执行以下命令，启动容器。
   ```
   docker run  -itd --name mrt_t --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g mrt_t /bin/bash
   ```

   其中：如果物理机上有数据集和权重，请添加`-v`参数用于将主机上的目录或文件挂载到容器内部，例如`-v <host_data_path>/<docker_data_path>`。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。

4. 执行以下命令，进入容器。
    ```
   docker exec -it mrt_t /bin/bash
   ```
5. 执行以下命令，启动虚拟环境。
   ```
   conda activate torch_env
   ```

### 2.4 启动训练

- 进入训练脚本所在目录
   ```
   cd /workspace/contrib/Pose/MRT
   ```



- 数据准备
    从[百度网盘](https://pan.baidu.com/s/1XguNfDk99aQcFOvcOVcoBw?pwd=3yv6)下载数据集文件夹，并按照如下目录结构组织文件和文件夹:
    ```
    .
    ├── MRT
    │   ├── Layers.py
    │   └── ...
    ├── mocap  # 训练，测试数据位置
    │   ├── discriminator_3_120_mocap.npy
    │   ├── test_3_120_mocap.npy
    │   └── train_3_120_mocap.npy
    ├── run_scripts # 启动
    │   ├── README.md
    │   ├── argument.py
    │   └── run_mrt.py
    ├── test_mrt.py
    ├── train_mrt.py
    └── ...
    ```


- 训练命令
    ```shell script
    # 单卡训练
    python run_scripts/run_mrt.py --device  sdaa --epoch 20 --batch_size 64 --data_path ./mocap --autocast
    # ddp训练
    python run_scripts/run_mrt.py --ddp --epoch 20 --batch_size 64 --data_path /root/MRT/mocap --autocast
    ```

- 验证命令
    ```shell script
    python run_scripts/run_mrt.py --device sdaa --checkpoint saved_model/19.model --data_path ./mocap --eval
    ```

- 复现结果
我们复现的预测结果为：

    |芯片|加速卡数量  | Epochs | 混合精度 |Batch size|avg 1 second|avg 2 seconds|avg 3 seconds|
    |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    |sdaa|1| 19 |是|64|0.97|1.55|2.10|

## License
This code is available for **non-commercial scientific research purposes** as defined in the [LICENSE file](./LICENSE). By downloading and using this code you agree to the terms in the [LICENSE](./LICENSE). Third-party datasets and software are subject to their respective licenses.
