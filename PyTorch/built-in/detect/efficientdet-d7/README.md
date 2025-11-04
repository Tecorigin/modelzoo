数据集路径：/mnt/nvme1/dataset/datasets/yolox
数据集预处理：
在完成数据集的摆放之后，我们需要利用voc_annotation.py获得训练用的2007_train.txt和2007_val.txt。   
修改voc_annotation.py里面的参数。第一次训练可以仅修改classes_path，classes_path用于指向检测类别所对应的txt。   
运行：
python train.py
### 性能情况
| 训练数据集 | 权值文件名称 | 测试数据集 | 输入图片大小 | mAP 0.5:0.95 |
| :-----: | :-----: | :------: | :------: | :------: |
| COCO-Train2017 | [efficientdet-d0.pth](https://github.com/bubbliiiing/efficientdet-pytorch/releases/download/v1.0/efficientdet-d0.pth) | COCO-Val2017 | 512x512 | 33.1 
| COCO-Train2017 | [efficientdet-d1.pth](https://github.com/bubbliiiing/efficientdet-pytorch/releases/download/v1.0/efficientdet-d1.pth) | COCO-Val2017 | 640x640 | 38.8  
| COCO-Train2017 | [efficientdet-d2.pth](https://github.com/bubbliiiing/efficientdet-pytorch/releases/download/v1.0/efficientdet-d2.pth) | COCO-Val2017 | 768x768 | 42.1
| COCO-Train2017 | [efficientdet-d3.pth](https://github.com/bubbliiiing/efficientdet-pytorch/releases/download/v1.0/efficientdet-d3.pth) | COCO-Val2017 | 896x896 | 45.6
| COCO-Train2017 | [efficientdet-d4.pth](https://github.com/bubbliiiing/efficientdet-pytorch/releases/download/v1.0/efficientdet-d4.pth) | COCO-Val2017 | 1024x1024 | 48.8
| COCO-Train2017 | [efficientdet-d5.pth](https://github.com/bubbliiiing/efficientdet-pytorch/releases/download/v1.0/efficientdet-d5.pth) | COCO-Val2017 | 1280x1280 | 50.2
| COCO-Train2017 | [efficientdet-d6.pth](https://github.com/bubbliiiing/efficientdet-pytorch/releases/download/v1.0/efficientdet-d6.pth) | COCO-Val2017 | 1408x1408 | 50.7 
| COCO-Train2017 | [efficientdet-d7.pth](https://github.com/bubbliiiing/efficientdet-pytorch/releases/download/v1.0/efficientdet-d7.pth) | COCO-Val2017 | 1536x1536 | 51.2  

### 所需环境
torch==1.2.0

### 文件下载  
训练所需的pth可以在百度网盘下载。       
包括Efficientdet-d0到d7所有权重。    
链接: https://pan.baidu.com/s/1cTNR63gTizlggSgwDrmwxg    
提取码: hk96    

VOC数据集下载地址如下，里面已经包括了训练集、测试集、验证集（与测试集一样），无需再次划分：  
链接: https://pan.baidu.com/s/1-1Ej6dayrx3g0iAA88uY5A    
提取码: ph32   
