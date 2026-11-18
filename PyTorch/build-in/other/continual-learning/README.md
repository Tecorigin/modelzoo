# continual-learning

## 1. 模型概述
continual-learning是一个基于 PyTorch 的持续学习（Continual Learning）算法综合实现库，由 Gido van de Ven 等人开发，用于复现其发表在 Nature Machine Intelligence（2022）上的论文《Three types of incremental learning》中的实验。
它系统性地实现了多种主流持续学习方法，包括 EWC、Synaptic Intelligence (SI)、LwF、iCaRL、Experience Replay (ER)、Deep Generative Replay (DGR) 等，并支持三种经典持续学习场景：任务增量（task）、领域增量（domain）和类别增量（class）。项目代码结构清晰，便于比较不同方法在 SplitMNIST、PermutedMNIST、CIFAR10/100 等基准数据集上的性能，是研究持续学习/灾难性遗忘问题的重要开源工具。


- 参考实现：
    ```
    url=https://github.com/GMvandeVen/continual-learning
    commit_id=e6d795aa81b9cef742b8de76cb71222d4d1ce00b
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

- 本实验采用MNIST数据集进行训练，请解压后放在仓库的store文件夹中。


### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/other/continual-learning
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_continual-learning`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_continual-learning
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_continual-learning -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_continual-learning /bin/bash
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
    cd /workspace/other/continual-learning
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2组织数据集。
  - 启动训练：
    ```
    python ./main.py --experiment=splitMNIST --scenario=task --si
    ```

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

MeanRelativeError: -0.121553406
MeanAbsoluteError: 0.00038663193
Rule,mean_relative_error -0.121553406
pass mean_relative_error=-0.121553406 <= 0.05 or mean_absolute_error=0.00038663193 <= 0.0002

