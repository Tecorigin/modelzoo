# YOLOv4-pytorch

## 1. 模型概述
 YOLOv4 目标检测算法的 PyTorch 复现版本，其核心结构（如 CSPDarknet53 骨干网络、PANet 特征融合、Mosaic 数据增强等）与原版论文和 Darknet 实现保持一致，但在框架移植过程中根据 PyTorch 特性进行了代码重构和模块化设计，便于训练、调试和部署。与原版（基于 C/CUDA 的 Darknet）相比，主要区别在于使用 PyTorch 框架实现，支持更灵活的训练流程、更易读的代码结构。


- 参考实现：
    ```
    url=https://github.com/argusswift/YOLOv4-pytorch
    commit_id=d34aaab4d03eb0f0121c97b951e2c9baa330794f
    ```


## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建Docker环境：介绍如何使用Dockerfile创建模型训练时所需的Docker环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考[基础环境安装](../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 准备数据集

- 可选用的开源数据集包括PascalVOC、MSCOCO 2017等，本实验采用coco2017数据集进行训练。

- 请你按以下结构组织COCO2017数据集：

   ```
    COCO
    ---train
    ---test
    ---val
    ---annotations
   ```
- 下载[权重](https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT)放在仓库的任意路径

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/Detection/YOLOv4
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_YOLOv4`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_YOLOv4
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_YOLOv4 -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_YOLOv4 /bin/bash
    ```

#### 2.3.5 执行以下命令，启动虚拟环境。

    ```
    conda activate torch_env_py310
    ```

#### 2.3.6 执行以下命令，安装其他环境依赖包。

    ```
    pip install -r requirements.txt
    ```


### 2.4 启动训练

#### 2.4.1 在Docker环境中，进入训练脚本所在目录。
    ```
    cd /workspace/Detection/YOLOv4
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2组织数据集。
  - 启动训练：
    ```
    python -u train.py  --weight_path weight/yolov4.weights
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

Parsed loss array (first 10): [1919.0723 1426.4371 1258.0116 1017.6067  863.7834  738.5779  652.9671
  581.8932  527.8826  479.3349]
Parsed loss array (first 10): [1970.9155 1600.3932 1385.5043 1110.7595  978.4315  833.1966  738.7197
  704.8448  633.0418  588.7084]
MeanRelativeError: -0.08819952
MeanAbsoluteError: -28.509356
Rule,mean_absolute_error -28.509356
pass mean_relative_error=-0.08819952 <= 0.05 or mean_absolute_error=-28.509356 <= 0.0002

