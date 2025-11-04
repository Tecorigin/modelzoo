# 模型推理适配指南

本文档介绍如何适配推理模型，使其能够在太初加速卡进行推理。整体流程如下：

1. 检查推理环境：使用Tecorgin ModelZoo进行推理适配前，检查Tecorgin ModelZoo的开发环境，确保您的开发环境能够充分满足当前任务需求。

2. 适配模型推理：对模型源码、推理相关接口等进行适配，使模型能够基于Tecorgin ModelZoo提供的环境进行推理。

3. 精度调试：如果适配后模型精度不达标，需要进行精度调试，确保其满足精度要求。

4. 性能调优：如果适配后的模型性能不达标，需要进行性能调优，确保其满足性能要求。

5. 添加Readme文件：基于适配的模型推理文件和代码，编写模型推理使用说明。

6. 添加模型的yaml信息：在[model.yaml](../contrib/model_config/model.yaml)中补充相关的参数设置，用于PR的功能性测试。

6. 提交PR：完成所有适配测试并通过后，将代码提交到Tecorigin ModelZoo仓库。



## 1. 检查推理环境

在使用Tecorgin ModelZoo进行推理适配之前，建议您先熟悉Tecorgin ModelZoo的开发环境配置，包括熟悉模型开发使用的框架、加速卡等硬件资源信息，以确保您的开发环境能够充分满足当前任务的需求，从而确保适配及推理过程的顺利进行。

本节介绍如何检查Tecorign ModelZoo的开发环境，包括硬件环境检查和容器环境检查。



### 1.1 检查硬件基本信息

使用`teco-smi`命令查看太初加速卡的硬件信息，了解当前可用的T1计算设备以及设备的工作状态。

```
(torch_env) root@DevGen03:/softwares# teco-smi
Wed Jun  5 02:46:48 2024
+-----------------------------------------------------------------------------+
|  TCML: 1.10.0        SDAADriver: 1.1.2b1        SDAARuntime: 1.1.2b0        |
|-------------------------------+----------------------+----------------------|
| Index  Name                   | Bus-Id               | Health      Volatile |
|        Temp          Pwr Usage|          Memory-Usage|             SPE-Util |
|=============================================================================|
|   0    TECO_AICARD_01         | 00000000:01:00.0     | OK                   |
|        35C                90W |        0MB / 15296MB |                   0% |
|-------------------------------+----------------------+----------------------|
|   1    TECO_AICARD_01         | 00000000:01:00.0     | OK                   |
|        35C                90W |      165MB / 15296MB |                   0% |
|-------------------------------+----------------------+----------------------|
|   2    TECO_AICARD_01         | 00000000:01:00.0     | OK                   |
|        35C                90W |      165MB / 15296MB |                   0% |
|-------------------------------+----------------------+----------------------|
|   3    TECO_AICARD_01         | 00000000:01:00.0     | OK                   |
|        35C                90W |      165MB / 15296MB |                   0% |
+-------------------------------+----------------------+----------------------+
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  Tcaicard     PID      Process name                            Memory Usage |
|=============================================================================|
|     1       76262      python3.8                                     165 MB |
|     2       76263      python3.8                                     165 MB |
|     3       76264      python3.8                                     165 MB |
+-----------------------------------------------------------------------------+
```

检查硬件信息时，您可以重点关注以下字段内容：

- Memory-Usage：T1计算设备内存使用状态。格式：使用内存 / 总内存。

- Health： T1计算设备的健康状态。``OK``表示T1计算设备运行正常；如果出现`DEVICE_LOST`、`HEARTBEAT_ERROR`等异常信息，请联系太初技术支持团队获取帮助。

- SPE-Util：T1计算设备计算核心SPE的使用率。如果出现`N/A`，表示T1芯片出现掉卡问题，请联系太初技术支持团队获取帮助。



### 1.2 检查容器软件信息

为便于您能够快速使用Tecorigin ModelZoo执行推理任务，Tecorigin ModelZoo当前以Docker容器的方式提供服务，Docker容器已经包含使用所需的所有基础软件及TecoinferenceEngine（小模型）推理框架。在安装Docker后，您可以按照以下步骤查看容器中的相关软件信息。

1. 在容器中，执行以下命令，查看Conda基础环境信息。

   ```
   (base) root@DevGen03:/softwares# conda info -e
   ```

   如果环境中包含`tvm-build`信息，表示基础环境正常。示例如下：

   ```
    # conda environments:
    base                  *  /root/miniconda3
    paddle_env               /root/miniconda3/envs/paddle_env
    torch_env                /root/miniconda3/envs/torch_env
    tvm-build                /root/miniconda3/envs/tvm-build
   ```

2. 进入Conda环境，执行以下命令，查看TecoinferenceEngine（小模型）框架及其依赖组件信息。

   ```
   (base) root@DevGen03:/softwares# conda activate tvm-build
   (tvm-build) root@DevGen03:/softwares# python -c "import tvm"
   ```

   如果终端成功输出TecoinferenceEngine（小模型）框架及其依赖组件的版本，表示TecoinferenceEngine（小模型）运行正常。示例如下：

   ```
   python -c "import tvm"

   # 输出以下内容
   ---------------+---------------------------------------------
   Host IP        | xx.xx.xx.xx
   ---------------+---------------------------------------------
   Teco-infer     | 1.2.0rc0+git8b22872
   TecoDNN        | 1.20.0
   TecoBLAS       | 1.20.0
   TecoCustom     | 1.20.0
   TECOCC         | 1.11.0
   SDAA Runtime   | 1.2.0
   SDAA Driver    | 1.2.0
   ---------------+---------------------------------------------

   ```



## 2. 模型推理适配

### 2.1 适配前准备

适配前需要准备的内容如下：

1. 检查并确认源码、模型和数据集。
2. 导出ONNX模型。
3. 适配标准。

#### 2.1.1 检查源码、模型和数据集

适配之前需要检查源码、模型和数据集，确保其满足适配要求。具体流程如下：

1. 根据以下源码优先级，选择合适的源码：太初指定的源码 > 官方源码 > 第三方开源 > 个人开源。

2. 确认适配模型的具体信息，例如：ResNet有多个系列，应明确适配哪一个版本、其输入shape和batch size信息。以适配ResNet50模型为例：`resnet50：input_shape:224x224，batch_size:1~128`。

3. 确认需要适配的数据集、模型权重和对应的metric指标，其中模型权重来源优先级：太初指定的权重 > 官方开源 > 第三方开源 > 个人开源 / 自己训练。

4. 在PyTorch/PaddlePaddle的CPU或GPU环境复现源码提供的metric指标，确保源码、模型和数据集的准确无误。



#### 2.1.2 导出ONNX模型

将PyTorch或PaddlePaddle框架上训练好的模型导出为ONNX格式的模型，推荐导出动态形状模型，导出后的模型需要使用`onnxsim`进行简化。

##### 2.1.2.1 安装ONNX依赖

导出ONNX格式的模型，需要安装一些ONNX依赖。具体依赖信息如下：

```
onnx>=1.12.0
onnxsim			# 用于简化模型
onnxruntime	    # 用于测试onnx推理
onnxconverter_common	# 用于将模型权重转为float16格式

# for torch model
torch>=1.12.0

# for paddle model
paddlepaddle
paddle2onnx
```

将以上依赖保存到`requirement.txt`文件中，然后使用下命令，进行安装：

```
pip install -r requirements.txt
```

##### 2.1.2.2 导出ONNX模型

- 从PyTorch导出ONNX格式模型

  以ResNet50模型为例，可参考如下代码导出ONNX格式模型：

  说明：将导出ONNX的代码保存为Python文件，后续需要提交到仓库中。

  ```
  import onnx
  import onnxsim		# 用于简化模型
  from onnxconverter_common import float16	# 用于将模型转为float16

  import torch
  import torchvision

  # init model
  resnet = torchvision.models.resnet50(pretrained=True)
  resnet.eval()

  # init dumpy_input
  dumpy_input = torch.randn(1, 3, 224, 224)

  # 静态shape导出
  torch.onnx.export(resnet,
                    dummy_input,
                    "resnet.onnx",
                    opset_version=12,         	# ONNX opset版本
                    input_names=['input'],    	# 输入名称
                    output_names=['output'],  	# 输出名称
                    do_constant_folding=True, 	# 是否执行常量折叠优化
                    dynamic_axes=None,	# 是否使用动态shape，不使用默认为None
                   )

  # 动态shape导出（推荐）
  dynamic_dims = {'input': {0: 'batch', 2: 'height', 3: 'width'},
                  'output': {0: 'batch'}}

  torch.onnx.export(resnet,
                    dummy_input,
                    "resnet_dyn.onnx",
                    opset_version=12,         	# ONNX opset版本
                    input_names=['input'],    	# 输入名称
                    output_names=['output'],  	# 输出名称
                    do_constant_folding=True, 	# 是否执行常量折叠优化
                    dynamic_axes=dynamic_dims,	# 是否使用动态shape，不使用默认为None
                   )

  # 以下为可选优化项，动态静态导出时均适用

  # Checks
  model_onnx = onnx.load("resnet_dyn.onnx")  # load onnx model
  onnx.checker.check_model(model_onnx)  # check onnx model

  # Simplify
  model_onnx, check = onnxsim.simplify(model_onnx)
  assert check, 'assert check failed'

  # convert_float_to_float16
  model_onnx = float16.convert_float_to_float16(model_onnx)
  onnx.save(model_onnx, "resnet_float16_dyn.onnx")
  ```

- 从PaddlePaddle导出ONNX格式模型

  以ResNet50模型为例，可参考如下代码导出ONNX格式模型：

  说明：将导出ONNX的代码保存为Python文件，后续需要提交到仓库中。

  ```
  import onnx
  import onnxsim		# 用于简化模型
  from onnxconverter_common import float16	# 用于将模型转为float16

  import paddle
  from paddle.vision.models import resnet50

  # init model
  model = resnet50(pretrained=True)

  # 静态shape
  input_spec = [
      paddle.static.InputSpec(
          shape=[1, 3, 224, 224], dtype="float32"),
  ]

  # 动态shape
  input_spec = [
      paddle.static.InputSpec(
          shape=[-1, 3, -1, -1], dtype="float32"),
  ]
  paddle.onnx.export(model,
                     "resnet.onnx",
                     input_spec=input_spec,
                     opset_version=12)

  # 以下为可选优化项，动态静态导出时均适用

  # Checks
  model_onnx = onnx.load("resnet.onnx")  # load onnx model
  onnx.checker.check_model(model_onnx)  # check onnx model

  # Simplify
  model_onnx, check = onnxsim.simplify(model_onnx)
  assert check, 'assert check failed'

  # convert_float_to_float16
  model_onnx = float16.convert_float_to_float16(model_onnx)
  onnx.save(model_onnx, "resnet_float16.onnx")
  ```

##### 2.1.2.3 ONNX导出常见问题

本节介绍导出ONNX模型时常见的问题以及相应的解决方法。

1. Conv动态权重问题

    导出ONNX模型时，若遇到[conv动态权重问题](https://github.com/pytorch/pytorch/issues/98497)，可以使用`torch.onnx.dynamo_export`接口导出ONNX（需要torch2.4.0及以上版本），具体可参考[PyTorch文档](https://pytorch.org/docs/2.4/onnx_dynamo.html#torch.onnx.dynamo_export)。

2. 算子不支持导致ONNX导出时报错

    导出ONNX模型时，若遇到算子不支持，导致导出ONNX报错，您可以在ONNX导出脚本中以自定义方式添加该算子，然后将其导出。 示例如下：

    ```python
    import torch
    import torch.nn as nn
    from torch.autograd import Function
    from torch.onnx import OperatorExportTypes
    class CustomAdd(Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, y: int):
            return x + y, x + y * 2
        @staticmethod
        def symbolic(g: torch.Graph, x: torch.Tensor, y: int):
            custom_node = g.op("custom::Add", x, y_i=y, outputs=2)
            return custom_node[0], custom_node[1]
    class MyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
        def forward(self, x: torch.Tensor):
            y = 1
            return CustomAdd.apply(x, y)
    def main():
        custum_model = MyModel()
        x = torch.randn(1, 3).to(torch.int64)
        torch.onnx.export(model=custum_model,
                            args=(x),
                            f="custom_add.onnx",
                            input_names=["x"],
                            output_names=['add1', 'add2'],
                            opset_version=17,
                            operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH)
    if __name__ == '__main__':
        main()
    ```


#### 2.1.3 适配标准

基线说明：模型推理的精度基线使用ONNXRuntime-CPU测试，性能基线使用TensorRT测试

- 推理精度：TecoInferenceEngine（小模型）推理的metric结果和ONNXRuntime-CPU的metric结果相对误差不超过0.1%；ONNXRuntime-CPU的metric结果与PyTorch/PaddlePaddle复现结果相对误差不超过1%。

- 极限性能：完成所有batch_size场景下的性能测试，除特别说明，性能不做要求。常用或源码默认的batch_size按照[1~max_batchsize]，逐渐增加batch size进行测试。例如：使用4个SPA时，shape中的batch_size为[1、4、8、16、......、512]。

##### 2.1.3.1 获取精度基线

1. 编写ONNXRuntime测试脚本，获取精度基线，可参考[ONNXRuntime文档](https://onnxruntime.ai/docs/get-started/with-python.html#quickstart-examples-for-pytorch-tensorflow-and-scikit-learn)。
2. 适配ModelZoo后使用pipeline获取，参考[适配推理Pipeline](#2224-适配推理pipeline)中`target`参数说明。

##### 2.1.3.2 获取性能基线

1. 可参考[GitHub: trtexec/README.md](https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec)获取`Throughput`和`Latency`性能数据。


### 2.2 适配模型推理

适配模型推理的主要流程如下：

1. ModelZoo目录说明：Fork ModelZoo仓库并按规范新建目录。
2. 适配ModelZoo推理接口：将源码中的数据集加载、数据预处理、后处理部分、模型推理代码抽出，按照ModelZoo的目录格式和推理要求进行适配。
3. 适配极限性能测试：随机初始化构造输入，获取模型特定`batch_size + shape`下的极限性能。
4. 检查适配后的推理模型是否满足要求。

#### 2.2.1 ModelZoo目录说明

#### 2.2.1.1 Fork ModelZoo仓库

基于Tecorigin ModelZoo仓库进行推理模型适配，首先需要您将[Tecorigin ModelZoo官方仓库](https://gitee.com/tecorigin/modelzoo)fork到您的个人空间，基于您的个人空间进行操作。关于如何Fork仓库，请查阅gitee官方使用文档：[Fork+PullRequest 模式](https://help.gitee.com/base/%E5%BC%80%E5%8F%91%E5%8D%8F%E4%BD%9C/Fork+PullRequest%E6%A8%A1%E5%BC%8F)。


#### 2.2.1.2 创建目录

Tecorigin ModelZoo小模型推理子仓`TecoInference`的`contrib`目录是贡献者推理相关代码的工作目录，其目录结构如下：

```
TecoInference/
├── contrib         # 注意，提交代码需要放在此目录内，其他目录下文件禁止修改。
│   ├── engine		# TecoinferenceEngine, 此处代码非必要禁止修改。
│   │   ├── base.py
│   │   ├── __init__.py
│   │   ├── tecoinfer_paddle.py     # paddle模型推理引擎模板
│   │   └── tecoinfer_pytorch.py    # torch模型推理引擎模板
│   ├── example		# 推理pipeline适配代码在该路径下，根据具算法方向选择文件夹或创建新文件夹
│   │   ├── classification
│   │   ├── detection
│   │	...
│   └── utils       # 数据集读取，预处理，后处理代码。
│       ├── datasets
│       │   └── __init__.py
│       ├── __init__.py
│       ├── postprocess
│       │   ├── __init__.py
│       │   ├── paddle
│       │   │   └── __init__.py
│       │   └── pytorch
│       │       └── __init__.py
│       └── preprocess
│           ├── __init__.py
│           ├── paddle
│           │   └── __init__.py
│           └── pytorch
│               └── __init__.py

```

在您本地的Tecorigin ModelZoo仓库中，新建一个目录`TecoInference/contrib/example/<算法领域>/<模型名称>`，用于存放适配后的推理相关代码。其中：

- 算法领域：当前有classification、detection、face、gnn、nlp、recommendation、reinforcement、segmentation和speech等，请您根据实际情况从中选择。
- 模型名称：对应的模型名称。

例如`GoogleNet`模型，其提交目录为：`TecoInference/contrib/example/classification/googlenet`。


#### 2.2.2 适配推理接口

将源码中的数据集加载、数据预处理、后处理、模型推理模块抽出，按照`TecoInference/contrib`目录格式和推理要求进行适配。

##### 2.2.2.1 适配数据集加载

从源码中抽出数据集加载相关代码进行适配，将适配后的代码保存成Python文件，然后将Python文件放在`TecoInference/contrib/utils/datasets`目录下。以ResNet模型为例，适配后的数据集加载代码如下：

```
import torch
import torchvision
from pathlib import Path
import os
import glob
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from functools import partial
import numpy as np
from torchvision.transforms.functional import InterpolationMode

RANK = int(os.getenv('RANK', -1))
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))

def fast_collate(memory_format, batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    return imgs, targets

def load_data(valdir, batch_size,rank=-1):
    # Data loading code
    print("Loading data")


    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
    )

    rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', rank))
    world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 0))

    print("Creating data loaders")
    if rank== -1:
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    else:
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False,num_replicas=world_size, rank=rank)


    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, sampler=test_sampler, num_workers=4, pin_memory=True,
        collate_fn=partial(fast_collate, torch.contiguous_format),shuffle=(test_sampler is None),
        drop_last=True ,
    )

    return data_loader_test
```

##### 2.2.2.2 适配预处理

从源码中抽出预处理相关代码进行适配，将适配后的代码保存成Python文件，然后将Python文件放在`TecoInference/contrib/utils/<框架>`目录下。其中：`框架`包含`paddle`和`pytorch`，请根据实际情况选择。

###### 2.2.2.2.1 预处理适配示例

以ResNet模型为例，适配后的预处理代码如下：

```
import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms


def process(img, resize_shape=256, crop_shape=224):
    img_transforms = transforms.Compose(
        [transforms.Resize(resize_shape), transforms.CenterCrop(crop_shape), transforms.ToTensor()]
    )
    img = img_transforms(img)

    with torch.no_grad():
        # mean and std are not multiplied by 255 as they are in training script
        # torch dataloader reads data into bytes whereas loading directly
        # through PIL creates a tensor with floats in [0,1] range
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img = img.float()

        input = img.unsqueeze(0).sub_(mean).div_(std)
    input = input.numpy()
    return input


def preprocess(image_path, dtype='float16', resize_shape=256, crop_shape=224):
    images = []
    if isinstance(image_path, str):
        if os.path.isfile(image_path):
            img = process(Image.open(image_path), resize_shape, crop_shape)
            images = [img]
        else:
            print("无法打开图片文件:", image_path)
            return None
    elif isinstance(image_path, Image.Image): #判断 Image 类型
        img = process(image_path, resize_shape, crop_shape)
        images = [img]
    elif isinstance(image_path[0],str): #判断 [str] 类型
        for i in image_path:
            img = process(Image.open(image_path), resize_shape, crop_shape)
            images.append(img)
    elif isinstance(image_path[0],Image.Image): #判断 [Image] 类型
        for i in image_path:
            img = process(i, resize_shape, crop_shape)
            images.append(img)
    else:
        print("输入有误")
        return None

    images = np.vstack(images)
    images = images.astype(np.float16) if dtype=='float16' else images.astype(np.float32)
    return images
```

###### 2.2.2.2.2 预处理适配常见问题

数据预处理阶段主要涉及输入数据形状以及数据类型处理问题，本节介绍输入数据形状以及数据类型问题的处理方法。

1. 输入形状处理

    模型推理时，如果输入数据的形状不满足预设形状需求，会出现输入形状（shape）相关的报错。为解决该类问题，需要在数据预处理时，对输入形状中不满足预设形状的维度进行padding。

    对维度进行padding，包含以下两个方面：

    - Batch维度padding：在未开启`drop_last`时，数据集迭代的最后一个batch数据可能不够组成预设batch size，需要对batch维度进行padding。Batch维度padding示例如下：

        ```python
        import torch
        import torch.nn.functional as F

        # 预设batch_size为32
        batch_size = 32

        # 假设有一个shape为[2, 3, 640, 640]的输入tensor
        images = torch.randn(2, 3, 640, 640)

        # 获取输入的batch
        count = images.shape[0]
        # 判断是否符合预设batch_size, 若不符合则需要padding
        if count < batch_size:
            # 对batch维度进行padding
            images_padded = F.pad(images, (0, 0, 0, 0, 0, 0, 0, batch_size - count))

        # 其他处理
        ......

        # 推理计算
        result = pipeline(images_padded)

        # 恢复真实数据的shape
        result = result[:count]
        ```
    - 其它维度padding：对于推理迭代时输入数据为变长的模型，需要将每次迭代的输入形状padding为预设形状（shape）。padding示例如下：

        ```python
        import torch
        import torch.nn.functional as F

        # 预设shape为[32, 3, 640, 640]
        image_shape = 640

        # 假设有一个shape为[32, 3, 640, 512]的输入tensor
        images = torch.randn(32, 3, 640, 512)

        # 将数据padding为shape[32, 3, 640, 640]
        images_padded = F.pad(images, (0, image_shape - images.shape[3], image_shape -images.shape[2], 0))

        # 其他处理
        ...

        # 推理计算
        result = pipeline(images_padded)

        # 计算后根据实际算法判断是否需要恢复数据shape进行后处理或输出
        ```

2. 输入数据处理

    TecoInferenceEngine（小模型）推理支持的输入数据为Numpy数组，需要在前处理阶段对输入数据的数据类型和连续性进行检查：

    - 数据类型检查：TecoInferenceEngine（小模型）推理的输入数据为Numpy数组，其数据类型应与使用ONNX转换的engine文件输入数据类型保持一致，否则会引起数据类型不一致的报错。
    - 连续性检查：输入的numpy数组应是连续数组。如果不连续，可以调用`np.ascontiguousarray`将非连续数组转为连续数组。示例如下：

        ```python
        import numpy as np
        if not model_input.flags.c_contiguous:	# 判断是否连续
            model_input = np.ascontiguousarray(model_input)	# 若不连续，进行处理
        ```


##### 2.2.2.3 适配后处理

从源码中抽出后处理相关代码进行适配，将适配后的代码保存成Python文件，然后将Python文件放在`TecoInference/contrib/utils/<框架>`目录下。其中：`框架`包含`paddle`和`pytorch`，请根据实际情况选择。

以ResNet模型为例，适配后的后处理代码如下：

```
import os
import numpy as np
from tvm.contrib.download import download_testdata


def postprocess(model_outputs, target='sdaa', topk=1):
    from scipy.special import softmax
    if os.path.exists('/mnt/checkpoint/TecoInferenceEngine/image_classification/synset.txt'):
        labels_path = '/mnt/checkpoint/TecoInferenceEngine/image_classification/synset.txt'
    else:
        labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
        labels_path = download_testdata(labels_url, "synset.txt", module="data")

    with open(labels_path, "r") as f:
        labels = [l.rstrip() for l in f]

    prec = []
    trt = True if target not in ['sdaa', 'cpu', 'onnx'] else False
    if trt:
        model_outputs = model_outputs.numpy()
    scores = softmax(model_outputs)
    scores = np.squeeze(scores)
    ranks = np.argsort(scores)[::-1]
    for rank in ranks[0:topk]:
        prec.append({'score':scores[rank],'label':labels[rank].split(' ',1)[1]})

    return prec
```

##### 2.2.2.4 适配推理Pipeline

推理pipeline适配主要包括`推理精度验证代码`、`单个样本推理代码`和`文件夹推理`三个部分：

- `推理精度验证代码`：基于数据集中的验证数据集进行推理，测试使用ONNXRuntime-CPU或TecoInferenceEngine进行模型推理的精度。
- `单个样本推理代码`：对单个图片或数据文件进行推理。
- `文件夹推理`：对文件中的所有文件进行推理。

注意：TecoinferenceEngine在太初卡上运行单卡三、四SPA推理时，会在每个SPA上初始化一份模型进行推理，因此模型初始化的`batch_size=单卡推理的batch_size/单卡SPA数量`。例如，当推理传入的ONNX文件的`batch_size`是16时，那么实际运行单卡四SPA推理时，实际传入的`batch_size`需要设置为64。

######  2.2.2.4.1 适配推理精度验证代码

在您创建的`TecoInference/contrib/example/<算法领域>/<模型名称>`目录下，创建`example_valid.py`文件，用于存放适配的推理精度验证代码。推理精度验证的关键代码及说明如下，完整的适配示例可参考ResNet模型的[example_valid.py](../example/classification/resnet/example_valid.py)。

```
# 添加engine和utils路径
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

# 导入推理引擎TecoInferEngine、优化文件路径PASS_PATH
from engine.tecoinfer_pytorch import TecoInferEngine
from engine.base import PASS_PATH

# 导入数据加载器load_data、预处理preprocess和后处理postprocess模块
from utils.datasets.image_classification_dataset import load_data
from utils.preprocess.pytorch.classification import preprocess
from utils.postprocess.pytorch.classification import postprocess

# 获取单卡三/四SPA环境变量
MAX_ENGINE_NUMS = int(os.getenv('MAX_ENGINE_NUMS', 4))	# 三/四SPA环境变量

# 添加最大推理step数
max_step = int(os.environ.get("TECO_INFER_PIPELINES_MAX_STEPS", -1))


if __name__ == "__main__":
    # 动态shape的onnx文件需要指定运行时模型的输入shape, 按照模型输入设置
    input_size = [[max(batch_size // MAX_ENGINE_NUMS, 1), 3, shape, shape]] # 参考3.3.2的注意内容。

    # 初始化模型，支持onnx/tensorrt/tvm
    pipeline = TecoInferEngine(ckpt=ckpt,				# 模型的onnx文件路径
                               input_name=input_name,	# 导出模型onnx时的input_name
                               target=target,			# 可选:'onnx'进行onnxruntime-cpu推理、'sdaa'进行TecoInferenceEngine推理
                               batch_size=batch_size,	# 数据集推理的batch_size
                               input_size=input_size,	# 指定初始化时的输入shape
                               dtype="float16", 		# 可选"float16"和"float32"，推荐"float16"
                               pass_path=pass_path,		# 推理框架优化文件，新适配模型设置为：PASS_PATH / "default_pass.py" 即可
                              )

    # load dataset
    val_loader = load_data(data_path, batch_size)

    # 统计性能
    e2e_time = []
    pre_time = []
    run_time = []
    post_time = []
    ips = []

    results = []
    # 遍历数据集进行推理，记录结果和性能数据
    for index, (input, target) in tqdm(val_loader):
        start_time = time.time()
        # 预处理, 需要将输入数据处理为np.ndarray或按照输入顺序处理为[np.ndarray, np.ndarray, ...]格式
        images = preprocess(input, dtype=opt.dtype)
        preprocess_time = time.time() - start_time

        # 进行推理，输出为numpy格式数据
        prec = pipeline(images)
        model_time = infer_engine.run_time

        # 后处理, 例如目标检测算法需要进行nms等
        result = postprocess(prec)
        infer_time = time.time() - start_time

        results.append(result)

        # 统计性能数据
        postprocess_time = infer_time - preprocess_time - model_time
        sps = batch_size / infer_time
        e2e_time.append(infer_time)
        pre_time.append(preprocess_time)
        run_time.append(pipeline.run_time)
        post_time.append(postprocess_time)
        ips.append(sps)
        if max_step > 0 and index >= max_step:
            break
    # metric计算，根据算法方向计算数据集推理的评价指标
    metric = get_acc(results)

    # 释放device显存，stream等资源
    if "sdaa" in opt.target:
        infer_engine.release()

    # 打印结果
    print('eval_metric', metric)
    print(f'summary: avg_sps: {np.mean(ips)}, e2e_time: {sum(e2e_time)}, avg_inference_time: {np.mean(run_time)}, avg_preprocess_time: {np.mean(pre_time)}, avg_postprocess: {np.mean(post_time)}')
```

###### 2.2.2.4.2适配单样本推理代码

在您创建的`TecoInference/contrib/example/<算法领域>/<模型名称>`目录下，创建`example_single_batch.py`文件，存放适配的单样本推理代码。单样本推理关键代码及说明如下，完整的适配示例可参考ResNet模型的[example_single_batch.py](../example/classification/resnet/example_valid.py)。

```
# 添加engine和utils路径
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

# 导入推理引擎TecoInferEngine、优化文件路径PASS_PATH
from engine.tecoinfer_pytorch import TecoInferEngine
from engine.base import PASS_PATH

# 导入数据加载器load_data、预处理preprocess和后处理postprocess模块
from utils.datasets.image_classification_dataset import load_data
from utils.preprocess.pytorch.classification import preprocess
from utils.postprocess.pytorch.classification import postprocess


if __name__ == "__main__":
    # 动态shape的onnx文件需要指定运行时模型的输入shape, 按照模型输入设置, 注意batch=1
    input_size = [[1, 3, shape, shape]]

    # 初始化模型，支持onnx/tensorrt/tvm
    pipeline = TecoInferEngine(ckpt=ckpt,				# 模型的onnx文件路径
                               input_name=input_name,	# 导出模型onnx时的input_name
                               target=target,			# 可选:'onnx'进行onnxruntime-cpu推理、'sdaa'进行TecoInferenceEngine推理
                               batch_size=batch_size,	# 数据集推理的batch_size
                               input_size=input_size,	# 指定初始化时的输入shape
                               dtype="float16", 		# 可选"float16"和"float32"，推荐"float16"
                               pass_path=pass_path,		# 推理框架优化文件，新适配模型设置为：PASS_PATH / "default_pass.py" 即可
                              )
    # 加载单个样本数据，并做预处理
    # 需要将输入数据处理为np.ndarray或按照输入顺序处理为[np.ndarray, np.ndarray, ...]格式
    input_data = load_data(demo_path, demo_infer=True)

    # 进行推理，输出为numpy格式数据
    prec = pipeline(images)

    # 后处理, 需要处理为可读的输出形式，例如目标检测算法打印坐标位置、种类和置信度, 分类模型打印出类别和score
    result = postprocess(prec)

    # 打印输出
    print(f"{demo_path}: {result}")
```

###### 2.2.2.4.3 适配文件夹推理代码

在您创建的`TecoInference/contrib/example/<算法领域>/<模型名称>`目录下，创建`example_multi_batch.py`文件，存放适配的文件夹推理代码。相较于单个样本推理，在单个样本推理的基础上添加文件遍历即可。文件夹推理的关键代码及说明如下，完整的适配示例可参考ResNet模型的[example_multi_batch.py](../example/classification/resnet/example_multi_batch.py)。

```
......
if __name__ == "__main__":

    ......

    for file_name in os.listdir(opt.data_path):
        file_path = os.path.join(opt.data_path, file_name)
        input_data = load_data(file_path, demo_infer=True)

        ......
```

#### 2.2.3 适配极限性能测试

极限性能测试通过随机初始化构造输入，获取模型特定`batch_size+shape`下的极限性能。适配极限性能测试包括适配极限性能执行脚本和极限性能测试配置信息。

- 适配极限性能执行脚本：在`TecoInference/contrib/tecoexec/testcase_configs`目录下新建`test_tecoexec.py`文件，用于存放极限性能执行代码。极限性能执行代码可参考[极限性能测试模板](../tecoexec/test_tecoexec.py)。
- 适配极限性能测试配置信息：在`TecoInference/contrib/tecoexec/testcase_configs`目录下新建`tecoexec_config.yaml`文件，用于存放极限性能测试配置信息。极限性能测试配置信息，请参考[极限性能测试配置模板](../tecoexec/testcase_configs/tecoexec_config.yaml)。

适配完成后，参考[文档](../tecoexec/README.md)完成功能测试。

#### 2.2.4 检查模型

适配完成后需要检查推理模型是否满足[适配标准](#213-适配标准)的要求。


### 3.精度调试

模型适配后，如果精度不能满足需求，则需要进行精度异常原因分析和调优。具体方法可以参考太初元碁官方文档[精度调测](http://docs.tecorigin.net/release/tecoinferenceengine/#6261b8696b0055e8a16199a0aeeb3f62)进行解决。


### 4.性能调优

模型适配后，如果训练性能不能满足需求，则需要进行性能分析和调优。具体方法可以参考太初元碁官方文档[性能调优](http://docs.tecorigin.net/release/tecoinferenceengine/#63ddfb2e68b756c19b91c94b0423334e)。


## 5. 添加README

基于适配的模型推理文件和代码，编写模型推理使用说明。文档格式可参考模板[resnet](https://gitee.com/tecorigin/modelzoo/blob/main/TecoInference/example/classification/resnet/README.md)，各章节需要严格对齐，必须包含以下内容：

```
# 算法名称
## 1. 模型概述
    对模型进行简介
## 2. 快速开始
    使用当前模型推理的主要流程，可直接复制模板。
### 2.1 基础环境安装
    使用当前模型推理的基础环境说明，可直接复制模板。
### 2.2 安装第三方依赖
    介绍第三方依赖安装，可直接复制模板, 注意修改模型路径。
### 2.3 获取ONNX文件
    提供导出onnx文件的方法，包括：权重下载、导出代码、导出命令相关参数说明
    注：如果需要, PyTorch或PaddlePaddle模型源码, 放在 export_onnx.py 或同级目录下。
### 2.4 获取数据集
    提供所用数据集下载链接和处理代码，确保用户可根据此处说明获取可用数据集。
### 2.5 启动推理
    提供单个样本和文件夹推理命令行，以及推理结果和推理参数说明（参考resnet/README.md）
### 2.6 精度验证
    提供数据集推理命令行、推理结果和推理结果说明（参考resnet/README.md）
```

## 6. 添加模型的yaml信息
用户在[model.yaml](../contrib/model_config/model.yaml)中补充相关的参数设置，用于PR的功能性测试。功能性测试包含两部分检测：

- 目录结构规范性检测：检查提交的模型目录下是否包含`README.md`，`requirements.txt`等必要文件。目录结构如下：

        └── model_dir
            ├──requirements.txt
            ├──README.md
            ...

- 模型功能性检查：根据用户提交的指令，检查onnx导出，数据集推理，单样本推理，多样本推理功能是否正常跑通，没有功能性错误。

yaml文件的具体信息参考[model yaml](../contrib/model_config/README.md)。


## 7. 提交PR

完成所有测试并通过后，您可以将代码提交到Tecorigin ModelZoo仓库。关于如何提交PR，参考[PR提交规范](https://gitee.com/tecorigin/modelzoo/blob/main/TecoInference/doc/PullRequests.md)。


