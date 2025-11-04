# 1、模型介绍

YOLOv4:YOLOv4是一种高效、准确的目标检测模型，采用无锚框设计和先进的数据增强技术，适用于实时应用。

# 2、快速开始

## 2.1基础环境安装

请参考[基础环境安装](https://gitee.com/tecorigin/modelzoo/blob/main/doc/Environment.md)章节，完成训练前的基础环境检查和安装。

## 2.2数据集的获取

### 2.2.1数据集介绍

本次训练使用的是PSACALVOC2012

### 2.2.2数据集的获取

https://www.alipan.com/t/9GhTweGgkbcIrY7GTVin

### 2.2.3数据准备：将数据集置于datasets文件夹下。结构目录如下所示。



    |-- yolov4-pytorch-master  
        |-- VOCdevkit         # 数据集名称
            |-- VOC2007          
                |-- Annotations          
                |-- JPEGImages             
                |-- ImageSets           
                    |-- Main           
    |--2007_train.txt
    |-- 2007_val.txt

## 2.3构建环境
### 1 执行以下命令，启动虚拟环境
```bash
conda activate teco-pytorch
```


### 2 安装所需要的包
```bash
pip install -r requirements.txt
```



## 2.4启动训练

### 1 在构建好的环境中，进入训练脚本所在目录
```bash
cd <ModelZoo_path>/PyTorch/contrib/Detection/yolov4-pytorch-master/run_scrips
```
### 2 训练指令
```bash
python ./run_scripts/run_yolov4.py
```
### 3 验证指令，利用训练的模型生成图像
```bash
python ./run_scripts/test.py
```





## 2.5 训练结果

| 加速卡数量    | 模型     | 混合精度           | batch_size | shape   | epoch | MAP   |
|----------|--------|----------------|------------|---------|-------|-------|
| 1 | yolov4 | AMP | 4          | 416*416 | 300   | 0.702 |