# MeshCNN

## 1. 模型概述
MeshCNN 是一个专为三维三角网格（3D meshes）设计的卷积神经网络框架，由 Rana Hanocka 等人在 SIGGRAPH 2019 提出。它直接在网格的边上定义卷积、池化和反池化操作，能够用于 3D 形状分类、分割等任务，其核心思想是通过学习对网格边进行重要性排序并逐步折叠（collapse），从而实现类似图像 CNN 中的空间下采样与特征提取。


- 参考实现：
    ```
    url=https://github.com/ranahanocka/MeshCNN.git
    commit_id=5bf0b899d48eb204b9b73bc1af381be20f4d7df1
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

- 本实验采用shrec_16数据集进行训练。

- 请你按以下结构组织shrec_16数据集：

   ```
    datasets
    ---shrec_16
        ---alien
        ---ants
        ---...
   ```
- 或者运行bash ./scripts/human_seg/get_data.sh下载训练所需的数据集。

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/Semantic_Segmentation/MeshCNN
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_MeshCNN`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_MeshCNN
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_MeshCNN -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_MeshCNN /bin/bash
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
    cd /workspace/Semantic_Segmentation/MeshCNN
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2组织数据集。
  - 启动训练：
    ```
    bash ./scripts/human_seg/train.sh
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

Parsed loss array (first 10): [3.39709  3.405514 3.399258 3.384965 3.407185 3.404497 3.423041 3.379833
 3.3561   3.384702]
Parsed loss array (first 10): [3.406057 3.419746 3.36483  3.43857  3.37713  3.366383 3.405066 3.372201
 3.406346 3.410691]
MeanRelativeError: -0.0060979356
MeanAbsoluteError: -0.020035299
Rule,mean_absolute_error -0.020035299
pass mean_relative_error=-0.0060979356 <= 0.05 or mean_absolute_error=-0.020035299 <= 0.0002

