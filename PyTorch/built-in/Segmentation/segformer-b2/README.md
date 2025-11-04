## SegFormer语义分割模型在Pytorch当中的实现
---
数据集路径：/mnt/nvme1/dataset/datasets/segformer
数据集预处理：
在完成数据集的摆放之后，需要利用voc_annotation.py获得训练用的2007_train.txt和2007_val.txt。   
修改voc_annotation.py里面的参数。第一次训练可以仅修改classes_path，classes_path用于指向检测类别所对应的txt。   

运行：
python train.py
### 所需环境
torch==1.2.0  

### 文件下载
训练所需的权值可在百度网盘中下载。     
链接: https://pan.baidu.com/s/1tH4wdGnACtIuGOoXb0_rAw    
提取码: tyjr      

VOC拓展数据集的百度网盘如下：   
链接: https://pan.baidu.com/s/1vkk3lMheUm6IjTXznlg7Ng    
提取码: 44mk    

### 训练步骤
#### a、训练voc数据集
1、将我提供的voc数据集放入VOCdevkit中（无需运行voc_annotation.py）。  
2、在train.py中设置对应参数，默认参数已经对应voc数据集所需要的参数了，所以只要修改backbone和model_path即可。  
3、运行train.py进行训练。  

#### b、训练自己的数据集
1、本文使用VOC格式进行训练。  
2、训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的SegmentationClass中。    
3、训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。    
4、在训练前利用voc_annotation.py文件生成对应的txt。    
5、在train.py文件夹下面，选择自己要使用的主干模型。
6、注意修改train.py的num_classes为分类个数+1。    
7、运行train.py即可开始训练。  

### 预测步骤
#### a、使用预训练权重
1、下载完库后解压，在百度网盘下载权值，放入model_data，修改segformer.py的backbone和model_path之后再运行predict.py，输入。  
```python
img/street.jpg
```
可完成预测。    
2、在predict.py里面进行设置可以进行fps测试、整个文件夹的测试和video视频检测。       
