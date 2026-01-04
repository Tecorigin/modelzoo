# VideoPose3D

## 1. 模型概述
VideoPose3D 是一种基于视频的单目3D人体姿态估计方法，它通过先使用2D姿态估计模型（如Mask R-CNN）从视频帧中提取2D关键点序列，再利用时空图卷积网络（如ST-GCN）或Transformer等时序模型对2D序列进行建模，直接回归出每一帧对应的3D人体关节点坐标。该方法采用自下而上的思路，分两阶段处理，既保证了2D检测的精度，又通过时序平滑和运动约束提升了3D姿态的稳定性和准确性，具有较好的实时性和鲁棒性，广泛应用于动作识别、虚拟现实和人机交互等领域。


- 参考实现：
    ```
    url=https://github.com/facebookresearch/VideoPose3D
    commit_id=1afb1ca0f1237776518469876342fc8669d3f6a9
    ```


## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建Docker环境：介绍如何使用Dockerfile创建模型训练时所需的Docker环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考[基础环境安装](../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 训练前准备

### 2.2.1 准备数据集
- 训练VideoPose3D模型，需要使用到Human3.6M数据集，由于原仓库中预处理后的数据集不再可用，一次需要转换原始数据集得到data_3d_h36m.npz和data_2d_h36m_gt.npz。
- 请从[Human3.6M官网](http://vision.imar.ro/human3.6m/)下载原始格式的数据集： Poses -> D3 Positions，然后将下载的文件解压至一个公共目录中，目录结构参考如下：
   ```
    /path/to/dataset/S1/MyPoseFeatures/D3_Positions/Directions 1.cdf
    /path/to/dataset/S1/MyPoseFeatures/D3_Positions/Directions.cdf
    ...
   ```
- 执行以下命令，运行数据预处理脚本：

   ```
    cd data
    python prepare_data_h36m.py --from-source-cdf /path/to/dataset
   ```

### 2.2.2 下载预训练模型
-  执行以下命令，下载预训练模型：
   ```
    mkdir checkpoint
    cd checkpoint
    wget https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_cpn.bin
    wget https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_humaneva15_detectron.bin
    cd ..
   ```

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/Video/VideoPose3D
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_VideoPose3D`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_VideoPose3D
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_VideoPose3D -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_VideoPose3D /bin/bash
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
    cd /workspace/Video/VideoPose3D
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2准备好数据集和预训练权重。
  - 启动训练：
    ```
    python run.py -e 80 -k cpn_ft_h36m_dbb -arc 3,3,3,3,3
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

Parsed loss array (first 10): [2.4229 2.8602 2.2217 1.6791 1.4892 1.3424 1.1691 1.0888 1.0589 0.9696]
Parsed loss array (first 10): [2.2952 2.8088 2.1314 1.6708 1.4117 1.2485 1.1333 1.0811 1.0178 0.9061]
MeanRelativeError: 0.0073278793
MeanAbsoluteError: 0.008313489
Rule,mean_relative_error 0.0073278793
pass mean_relative_error=0.0073278793 <= 0.05 or mean_absolute_error=0.008313489 <= 0.0002

