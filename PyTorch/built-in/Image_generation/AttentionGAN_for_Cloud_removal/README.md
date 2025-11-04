## 模型简介：
仓库围绕使用注意力机制的生成对抗网络（AttentionGAN）来实现云去除功能，该仓库的核心模型是基于注意力机制的生成对抗网络（AttentionGAN），用于借助合成孔径雷达（SAR）和光学图像实现云去除。其代码参考了 Attention - GAN 和 [CycleGAN](https://github.com/junyanz/pytorch - CycleGAN - and - pix2pix) 的实现。

## 数据集下载：
We use part of the data from the  [SEN12MS-CR dataset](https://patricktum.github.io/cloud_removal/sen12mscr/) in the paper.

Refer to our data：https://pan.baidu.com/s/12FodGp8xbnkLsq__1GCHlg, Extraction code: 1234.

You should organize your data into a format like this, replacing them with the data directory in this code:
```
data
│
└───train/test
│   │
│   └───trainA/testA  #cloud images
│   │   │   1.png
│   │   │   2.png
│   │   │   ...
│   │
│   └───trainB/testB  #cloudless images
│   │   │   1.png
│   │   │   2.png
│   │   │   ...
│   │
│   └───trainC/testC  #SAR images
│   │   │   1.png
│   │   │   2.png
│   │   │   ...
│ 
```

## 启动训练：
```shell
bash scripts/train.sh
```