#  ByteTrack
## 1. 模型概述
ByteTrack(ByteTrack: Multi-Object Tracking by Associating Every Detection Box) 是一个先进的多目标跟踪（Multi-Object Tracking, MOT）模型，旨在通过关联每一帧中的每一个检测框来实现高效的物体追踪。该方法在保持高精度的同时显著提升了处理速度，使其在实时应用中表现出色。通过关联每个检测框来跟踪，而不仅是关联高分的检测框。对于低分数检测框会利用它们与轨迹片段的相似性来恢复真实对象并过滤掉背景检测框。ByteTrack的核心在于其独特的轨迹关联策略，它不仅考虑了相邻帧之间目标的外观特征匹配，还结合了运动信息预测。具体来说，ByteTrack采用了一个轻量级的特征提取网络来获取每个检测框的嵌入向量，并利用卡尔曼滤波器预测目标在下一帧中的位置。通过这种方式，即使在复杂场景下也能有效地维持目标的身份连续性。

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
MOT-17数据集获取：
https://motchallenge.net/data/MOT17.zip

#### 2.2.2 处理数据集
python ByteTrack/dataset/mot/creat_json.py
处理后数据集结构：ByteTrack/dataset/mot/MOT17/
```angular2html
    |-- MOT17
        |-- annotations
            |-- train_half.json

        |-- images               
            |-- train  
                |--MOT17-02-DPM
                |--......
            |-- test
                |-- MOT17-01-DPM
                |-- ......
        |-- labels_with_ids
            |-- train
                |-- MOT17-02-DPM-- img1
                                    |--000001.txt
                                    |--......
                |-- ......
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
   sh train_sdaa_3rd.sh
   ```


### 2.5 训练结果

模型训练6h，得到结果如下  
|加速卡数量|模型|混合精度|Epoch|Batch size|Loss|  
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |   
|4|ByteTrack|是|10|8|2.125|
