#  FairMot
## 1. 模型概述
FairMOT 是一种多目标跟踪模型，能解决视频里目标检测与轨迹关联的问题。它采用统一的端到端架构，将目标检测和特征学习一起优化，避免分步处理导致的误差。模型用卷积神经网络骨干网络提取特征，引入高分辨率分支保留细节，提高对小目标和遮挡目标的跟踪能力。
它提出深度关联度量算法，综合考虑目标外观和运动信息，准确匹配不同帧目标。相比其他模型，FairMOT 不用额外后处理就能达到高精度，有效解决目标遮挡、轨迹断裂等难题。该模型在智能交通、安防监控、自动驾驶等领域广泛应用，助力交通分析、异常行为监测和行车安全保障，是多目标跟踪领域的重要成果。


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
airplane数据集获取：
wget https://bj.bcebos.com/v1/paddledet/data/mot/airplane.zip
下载解压存放于 dataset/mot目录下，并将其中的airplane.train复制存放于dataset/mot/image_lists
#### 2.2.2 处理数据集
python ByteTrack/dataset/mot/creat_json.py
处理后数据集结构：ByteTrack/dataset/mot/MOT17/
```angular2html
    |-- airplane
        |-- raw_videos
            |-- airplane-0
            |-- airplane-0.mp4
            |-- ......

        |-- images               
            |-- train  
                |--airplane-0
                |--......
            |-- workspace
        |-- labels_with_ids
            |-- train
                |--airplane-0
                |--......
        |-- airplane.train
        |-- airplane.half        
``` 


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
    cd <ModelZoo_path>/PaddlePaddle/contrib/Video/ByteTrack/script/
    ```
   
2. 运行训练。
   ```
   cd scripts
   ```

   ```
   sh train_sdaa_3rd.sh
   ```

### 2.5 训练结果

模型训练6h，得到结果如下  
|加速卡数量|模型|混合精度|Epoch|Batch size|Loss|  
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |   
|4|FairMot|是|40|6|7.2263|
