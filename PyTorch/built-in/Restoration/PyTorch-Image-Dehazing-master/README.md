PyTorch-Image-Dehazing-master
模型概述
PyTorch-Image-Dehazing-master 是一个基于 PyTorch 实现的图像去雾（Image Dehazing）项目。图像去雾是计算机视觉领域的一个重要任务，旨在从有雾的图像中恢复出清晰的图像。

2. 快速开始
使用本模型执行训练的主要流程如下：

基础环境安装：介绍训练前需要完成的基础环境检查和安装。
获取数据集：介绍如何获取训练所需的数据集。
构建Docker环境：介绍如何使用Dockerfile创建模型训练时所需的Docker环境。
启动训练：介绍如何运行训练。
2.1 基础环境安装
请参考基础环境安装章节，完成训练前的基础环境检查和安装。

2.2 准备数据集
2.2.1 获取数据集
/mnt/nvme1/dataset/datasets/PyTorch-Image-Dehazing-master/data
/mnt/nvme1/dataset/datasets/PyTorch-Image-Dehazing-master/original_image
2.3 构建Docker环境
使用Dockerfile，创建运行模型训练所需的Docker环境。

执行以下命令，进入Dockerfile所在目录。

cd <modelzoo-dir>/PyTorch/Classification/ResNet
其中： modelzoo-dir是ModelZoo仓库的主目录。

执行以下命令，构建名为sdaa_resnet50的镜像。

DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_resnet50
执行以下命令，启动容器。

docker run  -itd --name r50_pt -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_resnet50 /bin/bash
其中：-v参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的/datasets/目录。更多容器配置参数说明参考文档。

执行以下命令，进入容器。

docker exec -it r50_pt /bin/bash
执行以下命令，启动虚拟环境。

conda activate torch_env
2.4 启动训练
在Docker环境中，进入训练脚本所在目录。

cd /workspace/Classification/ResNet/run_scripts