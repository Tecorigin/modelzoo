# VGG16

## 1. 模型概述

VGG也称为VGGNet，是一种经典的卷积神经网络架构。VGG的开发是为了增加此类CNN的深度，以提高模型性能。它是具有多层的标准深度卷积神经网络架构。“深”是指由16个卷积层组成的VGG-16的层数。

## 2. 快速开始

使用本模型执行模型推理的主要流程如下：
1. 基础环境安装：介绍推理前需要完成的基础环境检查和安装。
2. 安装第三方依赖：介绍如何安装模型推理所需的第三方依赖。
3. 获取ONNX文件：介绍如何获取推理所需的ONNX模型文件。
4. 获取数据集：介绍如何获取推理所需的数据集。
5. 启动推理：介绍如何运行推理。
6. 精度验证：介绍如何验证推理精度。

### 2.1 基础环境安装

请参考推理首页的[基础环境安装](../../../../README.md)章节，完成推理前的基础环境检查和安装。

### 2.2 安装第三方依赖

1. 执行以下命令，进入conda环境。
   ```
   conda activate tvm-build
   ```

2. 执行以下命令，进入第三方依赖安装脚本所在目录。

   ```
   cd <modelzoo_dir>/TecoInference/contrib/classification/vgg16
   ```

3. 执行以下命令，安装第三方依赖。

   ```
   pip install -r requirements.txt
   ```

   **注意**：若速度过慢，可加上`-i`参数指定源。


### 2.3 获取ONNX文件
1. 下载权重文件，放到weight文件夹下。

[文件地址](https://download.pytorch.org/models/vgg16-397923af.pth)

2. 执行以下命令，导出模型的ONNX文件。

   
   ```
   python export_onnx.py
   ```
   
   此命令运行结束后会生成三个onnx文件，分别是vgg16.onnx，vgg16_dyn.onnx，vgg16_float16_dyn.onnx; 分别代表静态shape、动态shape、float16精度的动态shape格式的onnx。

### 2.4 获取数据集

您可以通过以下方式获取推理所需的数据集：
- 使用内置的demo数据样本。Demo数据样本位于仓库的`./images`目录。
- 使用[ImageNet数据集](/mnt/nvme/common/inference_dataset/imagenet/val)，用于模型推理和推理精度验证。

### 2.5 启动推理

1. 进入推理脚本所在目录。

   ```
   cd <modelzoo_dir>/TecoInference/contrib/classification/vgg16
   ```

2. 运行推理。

   - 单张图像推理：使用单张图像作为输入，进行推理。
   
     ```
     python example_single_batch.py  --ckpt vgg16_float16_dyn.onnx --batch_size 1 --input_name vgg16 --target sdaa --topk 1  --data_path './images/cat.png' --dtype float16
     ```

      推理结果：

      ```
      [{'score': 0.465, 'label': 'Egyptian cat'}]
      ```

    - 文件夹推理：使用文件夹，对文件中的所有图像进行推理。

      ```
      python example_multi_batch.py  --ckpt vgg16_float16_dyn.onnx --batch_size 1 --input_name vgg16 --target sdaa --topk 1  --data_path './images' --dtype float16
      ```

      推理结果：
  
      ```
      第0张 cat.png : [{'score': 0.465, 'label': 'Egyptian cat'}]
      第1张 hen.jpg : [{'score': 0.9854, 'label': 'hen'}]
      ```

 模型推理参数说明：

| 参数 | 说明 | 默认值 |
| ------------- | ------------- | ------------- |
| data_path  |数据路径 |./images/cat.png|
| ckpt       | 模型onnx路径  | N/A |
| batch_size | 推理的batch_size  | 1 |
| target     | 推理的设备 | `sdaa` |
| input_name |  onnx输入的名称 | vgg16  |
| dtype      | 模型推理的数据类型  | float16 |
| topk       |获取预测结果概率最大的前k个值计算准确率   | 1 |
| skip_postprocess | 跳过后处理  | True |

### 2.6 精度验证

执行以下命令，获得推理精度数据 float16。

```
python example_valid.py --ckpt vgg16_float16_dyn.onnx --batch_size 4 --input_name vgg16 --target sdaa --dtype float16 --topk 1 --data_path /mnt/nvme/common/inference_dataset/imagenet/val --skip_postprocess True
```

精度结果如下：

```
eval_metric 0.71466
summary: avg_sps: 203.79559156599768, e2e_time: 254.31298661231995, avg_inference_time: 0.005013230687477627, avg_preprocess_time: 0.01515904712677002, avg_postprocess: 0.00017020021438598633
```

执行以下命令，获得推理精度数据 float32。

```
python example_valid.py --ckpt vgg16_dyn.onnx --batch_size 16 --input_name vgg16 --target sdaa --dtype float32 --topk 1 --data_path /mnt/nvme/common/inference_dataset/imagenet/val --skip_postprocess True
```

精度结果如下：

```
eval_metric 0.71484
summary: avg_sps: 240.68964474981934, e2e_time: 213.42415857315063, avg_inference_time: 0.02172038035515027, avg_preprocess_time: 0.046331711044311524, avg_postprocess: 0.00024057106018066405
```

 结果说明：

| 参数 | 说明 |
| ------------- | ------------- |
| avg_sps | 吞吐量(images/s) |
| e2e_time | 端到端总耗时(s)  |
| avg_inference_time | 平均推理计算时间(s)  |
| avg_preprocess_time     | 平均预处理时间(s)  |
| avg_postprocess |  平均后处理时间(s) |
| eval_metric      | 数据集验证精度  |