#  Temporal Segment Network (TSN) 
## 1. 模型概述
Temporal Segment Network (TSN) 是视频分类领域经典的基于2D-CNN的解决方案。该方法主要解决视频的长时间行为识别问题，通过稀疏采样视频帧的方式代替稠密采样，既能捕获视频的全局信息，也能去除冗余，降低计算量。核心思想是将每帧的特征做平均融合作为视频的整体特征，再输入分类器进行分类。本代码实现的模型为基于单路RGB图像的TSN网络，Backbone采用ResNet-50结构。

## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装
请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集
#### 2.2.1 获取数据集
Kinetics-400数据集获取：
百度网盘链接：https://pan.baidu.com/s/17mNsY4pEwNZpgwl677eWxQ 提取码: nu78

#### 2.2.2 处理数据集
step1 合并分割的文件
  cat compress.tar.gz.* > compress.tar.gz

step2 解压合并后的文件,解压后，文件名为compress
  tar -xvzf compress.tar.gz

解压后Kinetics-400数据集结构：：
```angular2html
    |-- compress
        |-- train_256
            |-- abseiling
                ｜--xxx.mp4 
                ｜-- xxx.mp4 
            |-- air_drumming
                ｜--xxx.mp4
                ｜--xxx.mp4
            |-- ......
        |-- val_256               
            |-- abseiling  
                |--xxx.mp4
                |--xxx.mp4
            |-- air_drumming 
            |-- ......
```
step3 提取frames数据
在compress同级目录下，创建新目录： 
    ```
    mkdir k400
    ```

    ```
    cd compress
    ```

    ```
    mkdir train_256
    ```

    ```
    python /PaddleVideo/data/k400/extract_rawframes.py /data/datasets/compress/train_256/ /data/datasets/k400/train_256/ --level 2 --ext mp4
    ```

    ```
    mkdir val_256
    ```

    ```
    python /PaddleVideo/data/k400/extract_rawframes.py /data/datasets/compress/val_256/ /data/datasets/k400/val_256/ --level 2 --ext mp4
    ```

step4 获取训练集、验证集标签
https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/val_frames.list
https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/train_frames.list
将val_frames.list放入/data/datasets/k400/val_256/目录下、train_frames.list放入/data/datasets/k400/train_256/目录下


### 2.3 构建环境
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate paddle_env
    ```
   
2. 所需安装的包
    ```
    pip install -r requirements.txt
    ```
    
### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PaddlePaddle/contrib/Video/tsn/
    ```
   
2. 运行训练。

   ```
   export PADDLE_XCCL_BACKEND=sdaa
   ```
   ```
    python -B -m paddle.distributed.launch --gpus="0,1,2,3" --log_dir=log_tsm main.py  --amp --validate -c configs/recognition/tsm/tsm_k400_frames.yaml > "${LOG_FILE}" 2>&1
   ```

### 2.5 训练结果
|加速卡数量|模型|混合精度|Epoch|Batch size|Loss|  
| :-: | :-: | :-: | :-: | :-: | :-: |
|4|TSM|是|5|16|3.00|

