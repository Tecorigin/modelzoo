# groundingdinot

## 1. 模型概述

Grounding DINO-T 是一种用于视觉语言任务的先进模型，专注于目标定位和检测，特别是在文本描述引导下的目标检测任务中表现出色。它结合了 DINO（Detection Transformer）框架的强大检测能力和跨模态的语言理解能力，能够通过自然语言描述准确地在图像中定位目标。该模型在设计上采用了 Transformer 架构，支持端到端的训练方式，同时通过多层次的特征对齐和语言信息引导，实现了图像与文本的高效结合。Grounding DINO-T 的广泛应用包括远程感知、自动驾驶、智能监控等领域，特别适用于需要精确目标定位的多模态场景。

## 2. 快速开始

使用本模型执行模型推理的主要流程如下：
1. 基础环境安装：介绍推理前需要完成的基础环境检查和安装。
2. 安装第三方依赖：介绍如何安装模型推理所需的第三方依赖。
3. 获取ONNX文件和bert tokenizer：介绍如何获取推理所需的ONNX模型文件和bert tokenizer。
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
   cd <modelzoo_dir>/TecoInference/contrib/dedtection/groundingdinot
   ```

3. 执行以下命令，安装第三方依赖。

   ```
   pip install -r requirements.txt
   ```

   **注意**：若速度过慢，可加上`-i`参数指定源。


### 2.3 获取ONNX文件和bert tokenizer
1. 下载[权重文件](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)
，存放到权重目录中(例如/mnt/nvme/common/user_data/yqw/groundingdinot/weights/groundingdino_swint_ogc.pth)

2. 执行以下命令，导出模型的ONNX文件。

   
   ```
   python export_onnx.py --ckpt /mnt/nvme/common/user_data/yqw/groundingdinot/weights/groundingdino_swint_ogc.pth
   ```
   
   此命令运行结束后会生成三个onnx文件，分别是groundingdinot.onnx，groundingdinot_dyn.onnx，groundingdinot_dyn_float16.onnx.onnx; 分别代表静态shape、动态shape、float16精度的动态shape格式的onnx。在使用中请使用 groundingdinot_dyn_float16.onnx 进行推理。

3. bert tokenizer[下载地址](https://huggingface.co/google-bert/bert-base-uncased/tree/main)

### 2.4 获取数据集

您可以通过以下方式获取推理所需的数据集：
- 推理可使用内置的demo数据样本。Demo数据样本位于仓库的`./images`目录。
- 下载 COCO2017数据集 ([train](http://images.cocodataset.org/zips/train2017.zip), [val](http://images.cocodataset.org/zips/val2017.zip), [test](http://images.cocodataset.org/zips/test2017.zip)) and [labels](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-segments.zip). 


### 2.5 启动推理

1. 在Docker环境中，进入推理脚本所在目录。

   ```
   cd <modelzoo_dir>/TecoInference/example/detection/groundingdinot
   ```

2. 运行推理。

   - 单张图像推理：使用单张图像作为输入，进行推理。
   
     ```
     python example_single_batch.py  --ckpt ./groundingdinot_dyn.onnx --tokenlizer /mnt/nvme/common/user_data/yqw/groundingdinot/bert --batch_size 1 --input_name input --target onnx  --data_path ./images/bear.jpg --dtype float32
     ```

      推理结果：

      ```
      目标 1: - 类别: bear - 坐标: (0, 62, 639, 585) - 置信度: 0.82
      ```

    - 文件夹推理：使用文件夹，对文件中的所有图像进行推理。

      ```
      python example_multi_batch.py  --ckpt groundingdinot_dyn.onnx --tokenlizer /mnt/nvme/common/user_data/yqw/groundingdinot/bert --batch_size 1 --input_name input --target onnx --data_path ./images --dtype float32
      ```

      推理结果：
  
      ```
      第1张图片：bear.jpg
      目标 1: - 类别: bear - 坐标: (0, 62, 639, 585) - 置信度: 0.82

      第2张图片：room.jpg
      目标 1: - 类别: tv - 坐标: (3, 249, 102, 395) - 置信度: 0.78
      目标 2: - 类别: clock - 坐标: (298, 180, 307, 213) - 置信度: 0.74
      目标 3: - 类别: person - 坐标: (271, 236, 309, 444) - 置信度: 0.67
      目标 4: - 类别: vase - 坐标: (366, 451, 390, 602) - 置信度: 0.58
      目标 5: - 类别: tv - 坐标: (370, 311, 425, 431) - 置信度: 0.55
      目标 6: - 类别: wine glass - 坐标: (208, 287, 216, 320) - 置信度: 0.55
      目标 7: - 类别: chair - 坐标: (193, 327, 234, 478) - 置信度: 0.54
      目标 8: - 类别: potted plant - 坐标: (154, 264, 177, 319) - 置信度: 0.53
      目标 9: - 类别: chair - 坐标: (239, 329, 278, 479) - 置信度: 0.50
      目标 10: - 类别: person - 坐标: (255, 259, 266, 310) - 置信度: 0.49
      目标 11: - 类别: wine glass - 坐标: (240, 322, 248, 348) - 置信度: 0.46
      目标 12: - 类别: chair - 坐标: (228, 328, 276, 461) - 置信度: 0.43
      目标 13: - 类别: refrigerator - 坐标: (296, 252, 341, 430) - 置信度: 0.42
      目标 14: - 类别: dining table - 坐标: (307, 531, 425, 639) - 置信度: 0.41
      目标 15: - 类别: chair - 坐标: (202, 324, 250, 460) - 置信度: 0.38
      目标 16: - 类别: wine glass - 坐标: (110, 350, 124, 402) - 置信度: 0.37
      目标 17: - 类别: cup - 坐标: (110, 350, 124, 402) - 置信度: 0.36
      目标 18: - 类别: chair - 坐标: (196, 326, 234, 459) - 置信度: 0.35
      目标 19: - 类别: potted plant - 坐标: (226, 265, 253, 346) - 置信度: 0.35
      目标 20: - 类别: handbag - 坐标: (141, 450, 171, 492) - 置信度: 0.35
      目标 21: - 类别: vase - 坐标: (232, 322, 240, 347) - 置信度: 0.35
      目标 22: - 类别: vase - 坐标: (232, 310, 241, 348) - 置信度: 0.34
      目标 23: - 类别: dining table - 坐标: (198, 346, 298, 479) - 置信度: 0.33
      目标 24: - 类别: chair - 坐标: (249, 326, 292, 454) - 置信度: 0.33
      目标 25: - 类别: vase - 坐标: (159, 296, 168, 319) - 置信度: 0.30
      目标 26: - 类别: bottle - 坐标: (221, 283, 228, 321) - 置信度: 0.30
      目标 27: - 类别: cup - 坐标: (159, 296, 168, 319) - 置信度: 0.30
      目标 28: - 类别: cup - 坐标: (145, 8, 173, 59) - 置信度: 0.29
      目标 29: - 类别: microwave - 坐标: (316, 205, 350, 263) - 置信度: 0.28
      目标 30: - 类别: refrigerator - 坐标: (326, 259, 341, 428) - 置信度: 0.28
      目标 31: - 类别: bottle - 坐标: (264, 301, 271, 325) - 置信度: 0.27
      目标 32: - 类别: chair - 坐标: (251, 325, 285, 444) - 置信度: 0.27
      目标 33: - 类别: cup - 坐标: (311, 314, 319, 333) - 置信度: 0.26
      目标 34: - 类别: cup - 坐标: (192, 47, 217, 94) - 置信度: 0.26
      目标 35: - 类别: bottle - 坐标: (329, 231, 335, 260) - 置信度: 0.25
      目标 36: - 类别: chair - 坐标: (241, 327, 276, 456) - 置信度: 0.25
      目标 37: - 类别: bottle - 坐标: (326, 231, 331, 257) - 置信度: 0.25
      目标 38: - 类别: bottle - 坐标: (326, 230, 335, 259) - 置信度: 0.25
      目标 39: - 类别: cup - 坐标: (311, 275, 318, 295) - 置信度: 0.25
      目标 40: - 类别: bottle - 坐标: (305, 239, 311, 252) - 置信度: 0.25
      目标 41: - 类别: bottle - 坐标: (30, 320, 41, 362) - 置信度: 0.25
      目标 42: - 类别: dining table - 坐标: (0, 393, 144, 515) - 置信度: 0.25
      目标 43: - 类别: bottle - 坐标: (240, 322, 248, 348) - 置信度: 0.24
      目标 44: - 类别: cup - 坐标: (170, 32, 197, 77) - 置信度: 0.24
      目标 45: - 类别: microwave - 坐标: (341, 309, 350, 333) - 置信度: 0.24
      目标 46: - 类别: potted plant - 坐标: (2, 58, 115, 194) - 置信度: 0.22
      目标 47: - 类别: vase - 坐标: (221, 283, 228, 321) - 置信度: 0.22
      目标 48: - 类别: cup - 坐标: (170, 7, 197, 77) - 置信度: 0.21
      目标 49: - 类别: remote - 坐标: (60, 489, 79, 502) - 置信度: 0.21
      目标 50: - 类别: vase - 坐标: (110, 350, 124, 402) - 置信度: 0.21
      目标 51: - 类别: bottle - 坐标: (221, 282, 231, 323) - 置信度: 0.21
      目标 52: - 类别: bottle - 坐标: (225, 300, 231, 323) - 置信度: 0.20
      目标 53: - 类别: vase - 坐标: (365, 375, 402, 602) - 置信度: 0.20
      ```

 模型推理参数说明：

| 参数 | 说明 | 默认值 |
| ------------- | ------------- | ------------- |
| data_path  |数据路径 |./images/humandog.jpg|
| ckpt       | 模型onnx路径  | ./groundingdinot_dyn_float16.onnx |
| tokenlizer |tokenlizer路径 |/mnt/nvme/common/user_data/yqw/groundingdinot/bert|
| batch_size | 推理的batch_size  | 1 |
| input_size | 推理输入尺寸  | 800 |
| target     | 推理的设备 | `onnx` |
| input_name | onnx输入的名称 | input |
| dtype      | 模型推理的数据类型 | float32 |
| conf_thres | 推理结果的置信度阈值 | 0.2 |


### 2.6 精度验证

请提前准备好COCO数据集，执行以下命令，获得推理精度数据。

```
python example_valid.py --ckpt groundingdinot_dyn.onnx --tokenlizer /mnt/nvme/common/user_data/yqw/groundingdinot/bert --batch_size 1 --input_name input --target onnx --data_path /mnt/nvme/common/inference_dataset/coco/images/val2017 --dtype float32
```

精度结果如下：

```
IoU metric: 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.471
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.629
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.516
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.316
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.504
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.625
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.378
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.655
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.722
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.565
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.766
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.881
summary: avg_sps: 0.32664454940365234, e2e_time: 15456.114098787308, avg_inference_time: 3.03318007467268, avg_preprocess_time: 0.0023663184642791746, avg_postprocess: 0.05583997716903687
```

 结果说明：

| 参数 | 说明 |
| ------------- | ------------- |
| avg_sps | 吞吐量(images/s) |
| e2e_time | 端到端总耗时(s)  |
| avg_inference_time | 平均推理计算时间(s)  |
| avg_preprocess_time     | 平均预处理时间(s)  |
| avg_postprocess |  平均后处理时间(s) |
| IoU metric      | 数据集验证精度  |