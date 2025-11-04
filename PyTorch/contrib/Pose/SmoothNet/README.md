 # SmoothNet Training

## 1. 模型概述
[SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos (ECCV 2022)](https://github.com/cure-lab/SmoothNet)是人体姿态估计算法的后处理模型，旨在提高识别的人体姿态序列的精度和顺滑度。SmoothNet对于基于图像以及视频的人体姿态估计算法都有不错的提升效果，这里我们对[FCN体姿态估计算法](https://github.com/una-dinosauria/3d-pose-baseline)在[Human3.6M数据集](http://vision.imar.ro/human3.6m/description.php)中的识别人体姿态序列进行基于SmoothNet的优化复现。
<!-- toc -->

## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 构建Docker环境：介绍如何使用Dockerfile创建模型训练时所需的Docker环境。
3. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。

### 2.2 构建Docker环境

1. 执行以下命令，进入Dockerfile所在目录。
    ```
    cd <modelzoo-dir>/PyTorch/contrib/Pose/SmoothNet
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录

2. 执行以下命令，构建名为`smoothnet_t`的镜像。
   ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t  smoothnet_t
   ```

3. 执行以下命令，启动容器。
   ```
   docker run  -itd --name smoothnet_t --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g smoothnet_t /bin/bash
   ```

   其中：如果物理机上有数据集和权重，请添加`-v`参数用于将主机上的目录或文件挂载到容器内部，例如`-v <host_data_path>/<docker_data_path>`。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。

4. 执行以下命令，进入容器。
    ```
   docker exec -it smoothnet_t /bin/bash
   ```
5. 执行以下命令，启动虚拟环境。
   ```
   conda activate torch_env
   ```



### 2.3 启动训练
- 进入训练脚本所在目录
   ```
   cd /workspace/contrib/Pose/SmoothNet
   ```



- 数据准备

   从[百度网盘](https://pan.baidu.com/s/15zp3vtFQB_iLDei6MIKNGw?pwd=oihn)下载数据集data/文件夹。注意data文件夹较大，你可以选择下载，比如[FCN体姿态估计算法](https://github.com/una-dinosauria/3d-pose-baseline)+[Human3.6M数据集](http://vision.imar.ro/human3.6m/description.php)的数据集位置为data/poses/h36m_fcn_3D。

   最终文件夹结构应该如下:
   ```
   .
   ├── configs
   │   ├── h36m_fcn_3D.yaml
   │   └── ...
   ├── data
   |   ├── checkpoints         # SmoothNet官方提供的权重，可选择下载
   |   ├── poses               # 你需要的数据集，可选择下载
   |   |    ├── h36m_fcn_3D     
   |   |    ├── ...
   |   └── smpl               # SMPL格式姿态需要的数据，可选择下载
   ├── lib
   │   └── ...
   ├── requirements.txt
   ├── run_scripts    
   │   ├── README.md
   │   ├── argument.py
   │   └── run_smoothnet.py    # 启动脚本
   ├── train_smoothnet.py
   ├── eval_smoothnet.py
   ├── Dockerfile
   ├── LICENSE
   ├── NOTICE
   ├── README.md
   └── visualize_smoothnet.py
   ```

- 训练命令

    [FCN体姿态估计算法](https://github.com/una-dinosauria/3d-pose-baseline)+[Human3.6M数据集](http://vision.imar.ro/human3.6m/description.php)的训练命令：
    ```shell script
    python run_scripts/run_smoothnet.py --device sdaa --epoch 40 --batch_size 4000 --cfg configs/h36m_fcn_3D.yaml --dataset_name h36m --estimator fcn --body_representation 3D --slide_window_size 32 --autocast
    ```

- 训练结束验证

    [FCN体姿态估计算法](https://github.com/una-dinosauria/3d-pose-baseline)+[Human3.6M数据集](http://vision.imar.ro/human3.6m/description.php)的验证命令：
    ```shell script
    python run_scripts/run_smoothnet.py --device sdaa --cfg configs/h36m_fcn_3D.yaml --dataset_name h36m --estimator fcn --body_representation 3D --slide_window_size 32  --checkpoint results/path_to/epoch_23.pth.tar --eval
    ```

- 训练结果
    我们在[FCN体姿态估计算法](https://github.com/una-dinosauria/3d-pose-baseline)+[Human3.6M数据集](http://vision.imar.ro/human3.6m/description.php)组合中复现结果为

    |芯片|加速卡数量  | Epochs | 混合精度 |Batch size|slide_window_size| 吞吐量 | MPJPE | Accel |
    |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    |sdaa|1| 23 |是|4000|32| - | 52.70 | 1.10|
   