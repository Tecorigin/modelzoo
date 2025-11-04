# 1、模型介绍

cyclegan:CycleGAN是一种基于对抗生成网络的图像转换模型，不需要成对的训练数据，能够实现双向图像转换并保持图像的一致性。


# 2、快速开始

## 2.1基础环境安装

请参考[基础环境安装](https://gitee.com/tecorigin/modelzoo/blob/main/doc/Environment.md)章节，完成训练前的基础环境检查和安装。

## 2.2数据集的获取

### 2.2.1数据集介绍

本次训练使用的是病理学图像数据集，数据来自于ACROBAT 2023挑战赛的乳腺癌小型数据集。

### 2.2.2数据集的获取

https://www.alipan.com/s/6W6wageRZAv

### 2.2.3数据准备：将数据集置于datasets文件夹下。结构目录如下所示。



    |-- datasets  
        |-- HER21         # 数据集名称
            |-- trainA            # A域的训练集
            |-- trainB            # B域的训练集
            |-- testA             # A域的测试集
            |-- testB             # B域的测试集


## 2.3构建环境
### 1 执行以下命令，启动虚拟环境
```bash
conda activate torch_env
```


### 2 安装所需要的包
```bash
pip install -r requirements.txt
```



## 2.4启动训练

### 1 在构建好的环境中，进入训练脚本所在目录
```bash
cd <ModelZoo_path>/PyTorch/contrib/Image_generation/cycle/run_scrips
```
### 2 训练指令
```bash
python run_cyclegan.py --dataroot ../datasets/HER21 --name HER21_cyclegan --model cycle_gan --netG unet_128 --batch_size 1 --n_epochs 50 --n_epochs_decay 50
```
### 3 验证指令，利用训练的模型生成图像
```bash
python run_test.py --dataroot ../datasets/HER21 --name HER21_cyclegan --model cycle_gan --netG unet_128
```

### 4 评估指令，测试生成图像的质量

```bash
python evaluate.py --result_path <results saved path>
```
注意这里的路径要具体到image文件夹，例如：/root/cycle/run_scripts/results/HER21_cyclegan/test_latest/images


## 2.5 训练结果

| 加速卡数量    | 模型    | 混合精度           | batch_size |shape|epoch|psnr|ssim|
|----------|-------|----------------|------------|----------|-------|----------------|----------------|
| 1 | cycle_gan | AMP |    1        |256*256|100|22.91|0.44|
