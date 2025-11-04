# 1、模型介绍

YOLOX:YOLOX是一种高效、准确的目标检测模型，采用无锚框设计和先进的数据增强技术，适用于实时应用。

# 2、快速开始

## 2.1基础环境安装

请参考[基础环境安装](https://gitee.com/tecorigin/modelzoo/blob/main/doc/Environment.md)章节，完成训练前的基础环境检查和安装。

## 2.2数据集的获取

### 2.2.1数据集介绍

本次训练使用的是PSACALVOC2012

### 2.2.2数据集的获取

https://www.alipan.com/s/GtU6qr38Sd9

### 2.2.3数据准备：将数据集置于datasets文件夹下。结构目录如下所示。



    |-- datasets  
        |-- VOCdevkit         # 数据集名称
            |-- VOC2012          
                |-- Annotations          
                |-- JPEGImages             
                |-- ImageSets           
                    |-- Main           
                        |-- trainval.txt
                        |-- test.txt

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
cd <ModelZoo_path>/PyTorch/contrib/Detection/YOLOX-main/run_scrips
```
### 2 训练指令
```bash
python /root/YOLOX-main/run_scripts/run_yolox.py --model_name yolox --nnodes 1 -f exps/example/yolox_voc/yolox_voc_s.py  --conf 0.001

```
### 3 验证指令，利用训练的模型生成图像
```bash
python /root/YOLOX-main/run_scripts/run_test.py -f exps/example/yolox_voc/yolox_voc_s.py -d 0 -b 4
```





## 2.5 训练结果

| 加速卡数量    | 模型    | 混合精度           | batch_size |shape| epoch | MAP  |
|----------|-------|----------------|------------|----------|-------|------|
| 1 | yolox | AMP | 4          |256*256| 300   | 0.56 |