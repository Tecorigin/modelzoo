# 模型合入文件

## 1. 概述

在您完成模型训练代码开发或者模型训练代码适配后，欢迎您将相应代码贡献到Tecorigin ModelZoo开源社区。

本文档主要介绍贡献模型代码时需要准备的文件以及如何准备这些文件：

- 准备模型训练代码：在模型训练代码中添加注释和License，以及统一日志接口。
- 准备训练启动文件：创建模型训练代码的启动脚本。Tecorgin ModelZoo使用统一的规则启动模型训练，从而降低开发者的使用成本。
- 准备dockerfile文件：创建dockerfile文件，用于构建模型训练所需的Docker环境。
- 准备`requirement.txt`文件：创建`requirement.txt`文件，用于安装运行模型所需的第三方依赖。
- 准备Readme文件：编写Readme文件，介绍如何使用模型进行训练。


## 2. 准备模型训练代码

准备好模型训练代码后，在模型训练代码中添加注释和License，以及统一日志接口。
    
### 2.1 添加注释和License

#### 2.1.1 添加注释

对于代码中重要的部分，需要加入注释介绍功能，帮助用户快速熟悉代码结构，包括但不仅限于：
- 函数的功能说明。
- 变量的作用和意义。
- 重要的逻辑分支和关键点。

#### 2.1.2 添加License

在所有完全自主开发的代码文件和头文件内容最上方补充如下License内容：

- 对于C/C++文件需要添加C/C++ License：

  ```
  // BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
  // reserved.
  // Redistribution and use in source and binary forms, with or without
  // modification, are permitted provided that the following conditions are met:
  // Redistributions of source code must retain the above copyright notice,
  // this list of conditions and the following disclaimer.
  // Redistributions in binary form must reproduce the above copyright notice,
  // this list of conditions and the following disclaimer in the documentation
  // and/or other materials provided with the distribution.
  // Neither the name of the copyright holder nor the names of its contributors
  // may be used to endorse or promote products derived from this software
  // without specific prior written permission.
  //
  // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  // ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
  // LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
  // CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
  // SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  // INTERRUPTION)
  // HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
  // STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
  // WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
  // OF SUCH DAMAGE.
  ```
  
- 对于Python文件需要添加Python License：

  ```
  # BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
  # reserved.
  # Redistribution and use in source and binary forms, with or without
  # modification, are permitted provided that the following conditions are met:
  # Redistributions of source code must retain the above copyright notice,
  # this list of conditions and the following disclaimer.
  # Redistributions in binary form must reproduce the above copyright notice,
  # this list of conditions and the following disclaimer in the documentation
  # and/or other materials provided with the distribution.
  # Neither the name of the copyright holder nor the names of its contributors
  # may be used to endorse or promote products derived from this software
  # without specific prior written permission.
  #
  # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  # ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
  # LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
  # CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
  # SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  # INTERRUPTION)
  # HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
  # STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
  # WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
  # OF SUCH DAMAGE.
  ```

- 对已有第三方版权声明的原文件基础上进行修改的，以文件为单位增加注释说明，为减轻工作量，建议统一注释为：
  - C++：
    ```
    // Adapted to tecorigin hardware 
    ```
  - Python：
    ```
    # Adapted to tecorigin hardware。
    ```


### 2.2 添加统一日志接口

Tecorign ModelZoo使用[TCAP\_DLLogger](https://github.com/Tecorigin/tcap_dllogger)输出统一格式的log日志，以便于您更加直观地分析模型运行过程和结果。

在模型训练代码中添加logger日志整体上分为以下几步：

1. 在训练代码之前初始化logger。
   ```
   from tcap_dllogger import Logger, StdOutBackend,    JSONStreamBackend, Verbosity
   json_logger = Logger(
       [
           StdOutBackend(Verbosity.DEFAULT),
           JSONStreamBackend(Verbosity.VERBOSE,    'dlloger_example.json'),
       ]
   )
   ```
2. 定义需要输出的训练数据。在训练阶段至少需要包含`train.loss`和`train.ips`，在验证阶段需要包含`val.loss`和`val.ips`。如果你模型使用的并非`train.ips`衡量性能，也可以采用其他指标。
   ```
   json_logger.metadata("train.loss", {"unit": "", "GOAL":    "MINIMIZE", "STAGE": "TRAIN"})
   json_logger.metadata("train.ips",{"unit": "imgs/s",    "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})
   ```
3. 输出日志。
   ```
   json_logger.log(
   step = (epoch, step),
   data = {
           "rank":os.environ["LOCAL_RANK"],
           "train.loss":loss, 
           "train.ips":ips,
           },
   verbosity=Verbosity.DEFAULT,
   )
   ```
   **注意**：使用时，需要将以下代码添加到`requirements.txt`文件中。
   ```
   git+https://github.com/Tecorigin/tcap_dllogger.git
   ```

使用TCAP\_DLLogger后，输出的日志格式如下所示：
![image](./logger.png)

关于日志添加的详细代码，参考以下示例：
[https://gitee.com/xiwei777/modelzoo/commit/e8dabf0d3b1cebc2c13ecd1a125c9946ec2c342e](https://gitee.com/xiwei777/modelzoo/commit/e8dabf0d3b1cebc2c13ecd1a125c9946ec2c342e)

## 3. 准备训练启动文件

为简化及统一模型训练启动任务，降低开发者的使用成本，Tecorign ModelZoo使用统一的`run_scripts`接口规则运行模型。您需要创建`run_scripts`目录，统一管理模型运行接口。`run_scripts`目录下至少应当包括:

* `argument.py`文件：用于解析统一运行接口的参数，例如：`model_name`、`batchsize`、`epoch`等。
  
* `run_demo.py`文件：模型的运行脚本，启动脚本使用`os.system`或`subprocess`方法。

* `loss.py`文件：执行CUDA和SDAA Loss对比，生成评估数据和Loss曲线对比图。

* `test.sh`文件：对`loss.py`和`run_demo.py`的执行进行封装，快速运行模型和生成评估数据，便于模型验收。


关于`run_scripts`的详细信息，参考以下示例：https://github.com/Tecorigin/modelzoo/tree/acba72124121dda6acbfd485101a0d37bcbb32dc/PyTorch/contrib/Classification/Demo/run_scripts
    


## 4. 准备dockerfile文件

为降低使用成本，达到开箱即用，Tecorign ModelZoo所有模型基于Docker环境进行部署，Docker镜像环境使用dockerfile文件进行配置。提交模型时，您需要准备模型运行环境相关的dockerfile文件。

关于dockerfile的详细信息，参考以下示例：
- [TecoPaddle ResNet50 Dockerfile](https://github.com/Tecorigin/modelzoo/blob/main/PaddlePaddle/Classification/ResNet/Dockerfile)。
- [TecoPaddle Bert Dockerfile](https://github.com/tecorigin/modelzoo/blob/main/PyTorch/NLP/BERT/Dockerfile)


## 5. 准备requirement.txt文件

模型在训练时，需要安装第三方依赖。例如，ResNet50模型需要安装pillow、visualdl、tqdm等。您需要根据自己提供的模型类型，将模型相应的依赖写到`requirement.txt`文件中。

以PaddlePaddle框架训练的ResNet50模型为例，其[`requirments.txt`](https://github.com/tecorigin/modelzoo/blob/main/PaddlePaddle/Classification/ResNet/requirements.txt)内容如下：
```
prettytable
ujson
opencv-python==4.6.0.66
pillow
tqdm
PyYAML>=5.1
visualdl>=2.2.0
scipy>=1.0.0
scikit-learn>=0.21.0
gast==0.3.3
faiss-cpu
easydict
git+https://github.com/Tecorigin/tcap_dlloggerr.git
```

## 6. 准备Readme文件

Readme文件用于介绍模型的功能、环境依赖、数据集、使用方法、性能、精度等信息。README文件应包含如下内容：

* 模型概述：介绍模型，包含模型的来源和算法说明。
  
* Docker环境准备：包括准备基础环境、创建当前模型的Docker镜像、创建当前模型Docker容器和启动当前模型Docker容器等。
  
* 数据集：数据集获取及数据集处理方法。如果使用开源数据集或权重，提供开源获取方式和数据处理方法。如果使用非开源数据集或权重，请提供百度网盘下载链接和数据处理方法，数据集文件请不要提交代码上传，github单文件最大限制100MB。
  
* 启动训练的方法：包括进入对应启动脚本的目录和启动的方法。
  
* 训练结果: 使用表格提供简单的参数介绍，和对应的精度和性能结果。
  

Tecorign ModelZoo提供了[README](https://github.com/Tecorigin/modelzoo/blob/acba72124121dda6acbfd485101a0d37bcbb32dc/PyTorch/contrib/Classification/Demo/README.md)的Demo文件，您可以使用该Demo文件作为模板，将内容更换为提交模型相关的内容。

关于Readme的详细内容，参考以下示例：
[https://github.com/Tecorigin/modelzoo/blob/acba72124121dda6acbfd485101a0d37bcbb32dc/PyTorch/contrib/Classification/Demo/README.md](https://github.com/Tecorigin/modelzoo/blob/acba72124121dda6acbfd485101a0d37bcbb32dc/PyTorch/contrib/Classification/Demo/README.md)。
