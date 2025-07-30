# SSD

## 1. 模型概述

SSD（Single Shot MultiBox Detector）是一种用于目标检测的深度学习模型，专门用于在单张图像中同时检测多个对象并进行分类。SSD模型通过单次前向传播完成目标的边界框预测和分类任务，因而相比于传统的两阶段检测器（如Faster R-CNN）具有较高的速度。

## 2. 快速开始

使用本模型执行模型推理的主要流程如下：
1. 基础环境安装：介绍推理前需要完成的基础环境检查和安装。
2. 安装第三方依赖：介绍如何安装模型推理所需的第三方依赖。
3. 获取ONNX文件：介绍如何获取推理所需的ONNX模型文件。
4. 获取数据集：介绍如何获取推理所需的数据集。
5. 启动推理：介绍如何运行推理。
6. 精度验证：介绍如何验证推理精度。

### 2.1 基础环境安装

请参考推理首页的[基础环境安装](../../../README.md)章节，完成推理前的基础环境检查和安装。

### 2.2 安装第三方依赖

1. 执行以下命令，进入conda环境。
   ```
   conda activate tvm-build
   ```

2. 执行以下命令，进入第三方依赖安装脚本所在目录。

   ```
   cd <modelzoo_dir>/TecoInference/contrib/dedtection/ssd
   ```

3. 执行以下命令，安装第三方依赖。

   ```
   pip install -r requirements.txt
   ```

   **注意**：若速度过慢，可加上`-i`参数指定源。


### 2.3 获取ONNX文件
1. 下载[权重文件](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Flufficc%2FSSD%2Freleases%2Fdownload%2F1.2%2Fvgg_ssd300_coco_trainval35k.pth)
，存放到权重目录中(例如/mnt/nvme/common/user_data/yqw/ssd/mobilenet_v3_ssd320_voc0712.pth)

2. 执行以下命令，导出模型的ONNX文件。

   
   ```
   python export_onnx.py --ckpt /mnt/nvme/common/user_data/yqw/ssd/mobilenet_v3_ssd320_voc0712.pth
   ```
   
   此命令运行结束后会生成三个onnx文件，分别是ssd.onnx，ssd_dyn.onnx，ssd_dyn_float16.onnx.onnx; 分别代表静态shape、动态shape、float16精度的动态shape格式的onnx。在使用中请使用 ssd_dyn_float16.onnx 进行推理。

### 2.4 获取数据集

您可以通过以下方式获取推理所需的数据集：
- 推理可使用内置的demo数据样本。Demo数据样本位于仓库的`./images`目录。
- 下载Pascal VOC 2007数据集 ([下载链接](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)). 

### 2.5 启动推理

1. 在Docker环境中，进入推理脚本所在目录。

   ```
   cd <modelzoo_dir>/TecoInference/example/detection/ssd
   ```

2. 运行推理。

   - 单张图像推理：使用单张图像作为输入，进行推理。
   
     ```
     python example_single_batch.py  --ckpt ssd_dyn_float16.onnx --batch_size 1 --input_name input --target sdaa  --data_path ./images/humandog.jpg --dtype float16
     ```

      推理结果：

      ```
      目标 1: - 类别: person - 坐标: (10, -3, 323, 308) - 置信度: 0.98
      目标 2: - 类别: dog - 坐标: (40, 148, 209, 240) - 置信度: 0.30
      ```

    - 文件夹推理：使用文件夹，对文件中的所有图像进行推理。

      ```
      python example_multi_batch.py  --ckpt ssd_dyn_float16.onnx --batch_size 1 --input_name input --target sdaa --data_path ./images --dtype float16
      ```

      推理结果：
  
      ```
      第1张图片：ridehorse.jpg
      目标 1: - 类别: person - 坐标: (103, 44, 224, 178) - 置信度: 0.98
      目标 2: - 类别: horse - 坐标: (90, 106, 229, 282) - 置信度: 0.95

      第2张图片：humandog.jpg
      目标 1: - 类别: person - 坐标: (10, -3, 323, 308) - 置信度: 0.98
      目标 2: - 类别: dog - 坐标: (40, 148, 209, 240) - 置信度: 0.30
      ```

 模型推理参数说明：

| 参数 | 说明 | 默认值 |
| ------------- | ------------- | ------------- |
| data_path  |数据路径 |./images/humandog.jpg|
| ckpt       | 模型onnx路径  | ./ssd_dyn_float16.onnx |
| batch_size | 推理的batch_size  | 1 |
| input_size | 推理输入尺寸  | 320 |
| target     | 推理的设备 | `sdaa` |
| input_name | onnx输入的名称 | input |
| dtype      | 模型推理的数据类型 | float16 |
| conf_thres | 推理结果的置信度阈值 | 0.2 |


### 2.6 精度验证

请提前准备好Pascal VOC 2007数据集，执行以下命令，获得推理精度数据。

```
python example_valid.py --ckpt ssd_dyn_float16.onnx --batch_size 4 --input_name input --target sdaa --data_path /mnt/nvme/common/train_dataset/voc/VOC2007 --dtype float16
```

精度结果如下：

```
mAP@0.5~0.95:  0.6935122032657104
summary: avg_sps: 15.558954579681073, e2e_time: 319.1244738101959, avg_inference_time: 0.011535129794625856, avg_preprocess_time: 0.003789523499077472, avg_postprocess: 0.24244873184764906
```

 结果说明：

| 参数 | 说明 |
| ------------- | ------------- |
| avg_sps | 吞吐量(images/s) |
| e2e_time | 端到端总耗时(s)  |
| avg_inference_time | 平均推理计算时间(s)  |
| avg_preprocess_time     | 平均预处理时间(s)  |
| avg_postprocess |  平均后处理时间(s) |
| mAP@0.5~0.95      | 数据集验证精度  |