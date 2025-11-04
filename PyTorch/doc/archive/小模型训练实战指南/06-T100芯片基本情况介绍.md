# T1芯片基本情况介绍

## 概述

本教程介绍主要介绍T1芯片的基本信息，包括T1芯片的基本硬件信息以及与使用相关的软件基本信息。主要内容如下：

* 核心概念 
* 硬件信息
* 软件信息 
* 深度学习框架

## 核心概念

本节介绍T1芯片及太初软件栈使用相关的核心概念。

| **名称**          | **说明**                                                     |
| ----------------- | ------------------------------------------------------------ |
| T1芯片            | T1芯片负责处理大规模的数据运算，由计算核心阵列SPA（Synergistic Processor Element Array）和对应的Global高带宽存储器构成。 |
| 计算核心阵列SPA   | 每个计算核心阵列SPA是一个协从运算单元，由计算核心SPE（Synergistic Processor Element）构成。 |
| 计算核心SPE       | 计算核心SPE相当于一个线程，您可以使用SPE进行异构并行计算。   |
| SDAA后端/SDAA设备 | 在TecoPytorch/TecoPaddle框架中， SDAA后端特指T1计算设备，在不同上下文中会使用SDAA设备替代。 |
| TCCL              | TCCL（Tecorigin Collective Communication Library，太初集合通信库）是T1芯片通信的高性能集合通信库。通过API接口便捷地管理多个T1芯片间的集合通信操作，从而实现高效的数据并行处理，有力支持AI训练场景的集合通信操作。 |
| TecoSMI           | TecoSMI是T1芯片的监控和管理的命令行工具，为用户提供T1芯片的硬件信息和状态，例如设备属性、设备内存、核心SPE使用率等重要信息，帮助用户快速发现异常信息、提升问题诊断效率、协助上层用户进行高效开发和调试。 |

## 硬件信息

### T1芯片构成

1芯片由计算核心阵列SPA（Synergistic Processor Element Array）和对应的Global高带宽存储器构成。
每个计算核心阵列SPA是一个协从运算单元，由计算核心SPE（Synergistic Processor Element）构成。
![image](C:/Users/00489/Desktop/modelzoo/ModelZoo生态教程/模型适配及PR合入0801/T1芯片.png)

### T1芯片基本信息

T1芯片的一张实体的物理卡上有多个SPA，实际使用时会被抽象成多个虚拟的device，简单来说，一张实体卡=多张虚拟卡。您可以使用`teco-smi`命令，查看T1芯片的硬件信息和状态。

```
teco-smi
+-----------------------------------------------------------------------------+
|  TCML: x.x.x          SDAADriver: x.x.x          SDAARuntime: x.x.x         |
|-------------------------------+----------------------+----------------------|
| Index  Name                   | Bus-Id               | Health      SPE-Util |
|        Temp          Pwr Usage|          Memory-Usage|                      |
|=============================================================================|
|   0    TECO_AICARD_01         | 00000000:22:00.0     | OK                0% |
|        0C                  0W |        0MB / 15296MB |                      |
|-------------------------------+----------------------+----------------------|
|   1    TECO_AICARD_01         | 00000000:22:00.0     | OK                0% |
|        0C                  0W |        0MB / 15296MB |                      |
|-------------------------------+----------------------+----------------------|
|   2    TECO_AICARD_01         | 00000000:22:00.0     | OK                0% |
|        0C                  0W |        0MB / 15296MB |                      |
|-------------------------------+----------------------+----------------------|
|   3    TECO_AICARD_01         | 00000000:22:00.0     | OK                0% |
|        0C                  0W |        0MB / 15296MB |                      |
|-------------------------------+----------------------+----------------------|
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  Tcaicard     PID      Process name                            Memory Usage |
|=============================================================================|
|     3     1230779      sdaa_runtime_te                                 2 MB |
+-----------------------------------------------------------------------------+
```

上面结果中，`Index`字段是device的编号，您可以看到该T1芯片包含4个device，每个device提供了15+G的显存空间。

如果想用到一张卡上所有的算力，可以使用DDP分布式计算。


## 软件信息

### Docker环境

当前Tecorign ModelZoo提供的Docker环境，容器已经包含使用所需的所有基础软件及深度学习框架TecoPyTorch 和TecoPaddle，开箱即用。

当前太初框架框架提供的Docker环境，包含torch和paddle的conda虚拟环境，您可以直接使用Docker环境进行模型训练。

```
(base) root@DevGen03:/softwares# conda info -e 
# conda environments:
#
base                  *  /root/miniconda3
paddle_env               /root/miniconda3/envs/paddle_env
torch_env                /root/miniconda3/envs/torch_env
tvm-build                /root/miniconda3/envs/tvm-build
```

### 框架及依赖组件信息

您可以通过以下方式，查看Docker中的框架及依赖组件信息，以TecoPyTorch为例：

 ```
(base) root@DevGen03:/softwares# conda activate torch_env  
(torch_env) root@DevGen03:/softwares# python -c "import torch,torch_sdaa"
--------------+------------------------------------------------
 Host IP      | N/A
 PyTorch      | 2.0.0a0+gitdfe6533
 Torch-SDAA   | 1.6.0b0+git19f8ed9
--------------+------------------------------------------------
 SDAA Driver  | 1.1.2b1 (N/A)
 SDAA Runtime | 1.1.2b0 (/opt/tecoai/lib64/libsdaart.so)
 SDPTI        | 1.1.0 (/opt/tecoai/lib64/libsdpti.so)
 TecoDNN      | 1.19.0b3 (/opt/tecoai/lib64/libtecodnn.so)
 TecoBLAS     | 1.19.0b3 (/opt/tecoai/lib64/libtecoblas.so)
 CustomDNN    | 1.19.0b1 (/opt/tecoai/lib64/libtecodnn_ext.so)
 TecoRAND     | 1.6.0b0 (/opt/tecoai/lib64/libtecorand.so)
 TCCL         | 1.16.0b0 (/opt/tecoai/lib64/libtccl.so)
-------
 ```

建议非必要情况下，不要轻易更改Torch-SDAA和Paddle-SDAA的版本。

## 深度学习框架

太初针对T1芯片适配的深度学习框架包括TecoPyTorch和TecoPaddle。

*   TecoPyTorch：

    * 在线课程：[http://docs.tecorigin.com/release/tecopytorch\_course/](http://docs.tecorigin.com/release/tecopytorch_course/)

    * 文档：[http://docs.tecorigin.com/release/tecopytorch/](http://docs.tecorigin.com/release/tecopytorch/)

*   TecoPaddle

    * 在线课程：[http://docs.tecorigin.com/release/tecopaddle\_course](http://docs.tecorigin.com/release/tecopaddle_course)

    * 文档：[http://docs.tecorigin.com/release/tecopaddle](http://docs.tecorigin.com/release/tecopaddle)

