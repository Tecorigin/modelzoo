# vgg16

## 模型简介：
vgg16是2014年ImageNet比赛的亚军，也是当年ImageNet上的冠军，vgg16的结构如下：
是2014年ImageNet比赛的亚军，也是当年ImageNet上的冠军，的结构如下：
![](URL_ADDRESS![vgg16](https://github.com/chenyuntc/pytorch-cifar100/blob/master/imgs/vgg16.png))  
vgg16的网络结构中，卷积层的参数都是一样的，只有全连接层的参数不一样。

## 数据集下载：
模型使用cifar100作为训练数据集，训练过程中会自动下载。

## 开始训练：
python train.py -net vgg16 -gpu

## 开始测试：
python test.py -net vgg16 -weights path_to_vgg16_weights_file