# Conv-VAE-PyTorch

## 模型简介：
标准变分自动编码器 （VAE） 的 PyTorch 实现。摊销推理模型（编码器）由卷积网络参数化，而生成模型（解码器）由转置卷积网络参数化。近似后验的选择是具有对角协方差的完全分解高斯分布。

## 数据集准备：
wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip
unzip celeba.zip

## 模型训练：
要开始训练模型，请修改config.json文件，然后运行以下命令：
```
python train.py --config config.json
```

## 模型测试：
要测试模型，请运行以下命令：
```
python test.py --resume path/to/checkpoint
```
