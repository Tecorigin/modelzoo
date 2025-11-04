# Onsets and Frames

## 模型概述
Onsets and Frames是由Google Brain团队研制的一种复音钢琴自动转录模型，具有比之前的同类模型更大的参数和更加精准的预测结果。Onsets and Frames在众多SVC（Singing Voice Conversion）以及SVS（Singing Voice Synthesis）任务中可以作为优秀的midi输入源，因此Onsets and Frames已经成为众多语音界模型如  韵泉唱（YQ maintained diffsinger）、地府唱（Diffsinger）、大市唱（AllSing）的midi提供模型之一。同时模型的midi预测输出也可以作为音乐宿主软件（DAW，Digital audio workstation。如 Cubase、 FL Studio等）的乐器轨道的音符输入，为音乐创作者提供了便利。

## Quick Start Guide

使用本模型准备训练的流程如下：
1. 基础环境安装：介绍了如何安装基础的环境和依赖
2. 获取数据集：介绍了如何获取数据集
3. 启动训练：介绍了如何启动训练

### 1、环境准备



#### 1.1 拉取代码仓

``` bash
git clone https://gitee.com/tecorigin/modelzoo.git
```

#### 1.2 Teco环境准备

``` bash
cd /modelzoo/PaddlePaddle/contrib/Speech/Onsets-and-Frames
conda activate paddle_env

# 执行以下命令验证环境是否正确，正确则会打印如下版本信息
python -c "import paddle"

>>> sdaa plugin compiled with gcc
>>> PaddlePaddle Compilation Commit: 733b5d7f4e8680f0eb8fe4e8d371cfffadd4a3fd
>>> PaddlePaddle Compilation Version: 2.5.1
>>> +---------+--------+---------------+--------------+-------------+----------+-----------+-----------+-------------+-------+--------------------+
>>> |         | paddle | paddle_commit | sdaa_runtime | sdaa_driver | teco_dnn | teco_blas | teco_tccl | teco_custom | sdpti | paddle_sdaa_commit |
>>> +---------+--------+---------------+--------------+-------------+----------+-----------+-----------+-------------+-------+--------------------+
>>> | Version | 2.5.1  | 2e24fe5       | 1.0.0        | 1.0.0       | 1.17.0b0 | 1.17.0b0  | 1.14.0    | 1.17.0b0    | 1.0.0 | 04c5143            |
>>> +---------+--------+---------------+--------------+-------------+----------+-----------+-----------+-------------+-------+--------------------+
```

#### 1.3 安装依赖

```bash
pip install -r requirements.txt
```

### 2、准备数据集
#### 2.1 数据集说明

Onsets and Frames支持MAPS和Maestro数据集。

MAPS数据集网上不是很好找，建议Fork在AI Studio上面的[项目](https://aistudio.baidu.com/projectdetail/6722378 "Onsets and Frames项目")获取其中的MAPS数据集，或者在工作路径下联网运行`run_scripts/downloadMAPS.py`文件即可正确下载并放置数据集。

如果想要使用Maestro数据集，您需要准备至少24G显存和64G内存。

执行datas目录下`prepare_maestro.sh`文件以下载并处理Maestro数据集。

不论是MAPS数据集还是Maestro数据集，数据集都应该放在datas目录下。

#### 2.2 数据集目录结构

在datas目录下按照该数据结构放置数据文件夹：

```
└── datas
    ├──MAPS
    │  │
    │  └── ... 
    │
    └──Maestro
       │
       └── ...
```


其中train文件夹下为用于训练的数据，val文件夹下为测试的数据。data文件夹下已经有预先放置好的示例数据集。



### 3、 启动训练

举个例子：

```Bash
python run_scripts/train.py -ci 100 -bs 4 -l model -ri 300000 -lr 6e-4 -i 301300
```

具体的启动参数请参考run_scripts目录下的自述文件。

#### 3.1 启动模型

有些算子在sdaa设备上面是不支持的，所以需要设置算子黑名单环境变量：
```bash
# conv2d不支持group非1，因为反向传播还有个单独的算子，所以需要再来个conv2d_grad
export CUSTOM_DEVICE_BLACK_LIST=conv2d,conv2d_grad
```

使用单卡训练：

```bash
python run_scripts/train.py --logdir runs/model --iterations 1000000
```

或者试试多卡训练，首先需要配置好相关的环境变量：
参考相关的[issue](https://gitee.com/tecorigin/modelzoo/issues/I9S4CZ) 
```bash
export PADDLE_XCCL_BACKEND=sdaa
export SDAA_VISIBLE_DEVICES=0,1,2,3
```

开始使用DDP多卡训练

```bash
python run_scripts/train.py -ci 100 -bs 4 -l model -ri 300000 -lr 6e-4 -i 301300 -npn 4
```

### 4、验证模型效果和推理数据

对已经训练好的模型进行MAPS数据集上面的验证：

```Bash
# 设备必须使用cpu，否则算子会有不兼容
python run_scripts/evaluate.py model-300000.pdparams --device cpu
```

使用已经训练好的模型进行推理，转录MIDI，为各种音乐宿主软件或者模型提供MIDI输入：

```Bash
# 举个例子
python run_scripts/transcribe.py model-300000.pdparams Illusionary_Daytime-Shirfine-27983780.flac
```

其中，使用Maestro数据集训练的Onsets-and-Frames模型在MAPS数据集上的指标为：

```Text
                            note precision                : 0.845 ± 0.081
                            note recall                   : 0.824 ± 0.098
                            note f1                       : 0.834 ± 0.087
                            note overlap                  : 0.704 ± 0.079
               note-with-offsets precision                : 0.592 ± 0.135
               note-with-offsets recall                   : 0.577 ± 0.139
               note-with-offsets f1                       : 0.584 ± 0.136
               note-with-offsets overlap                  : 0.840 ± 0.077
              note-with-velocity precision                : 0.803 ± 0.085
              note-with-velocity recall                   : 0.782 ± 0.097
              note-with-velocity f1                       : 0.792 ± 0.089
              note-with-velocity overlap                  : 0.706 ± 0.078
  note-with-offsets-and-velocity precision                : 0.567 ± 0.135
  note-with-offsets-and-velocity recall                   : 0.553 ± 0.138
  note-with-offsets-and-velocity f1                       : 0.560 ± 0.136
  note-with-offsets-and-velocity overlap                  : 0.840 ± 0.078
                           frame f1                       : 0.779 ± 0.066
                           frame precision                : 0.832 ± 0.066
                           frame recall                   : 0.736 ± 0.080
                           frame accuracy                 : 0.643 ± 0.088
                           frame substitution_error       : 0.064 ± 0.026
                           frame miss_error               : 0.200 ± 0.066
                           frame false_alarm_error        : 0.086 ± 0.047
                           frame total_error              : 0.350 ± 0.097
                           frame chroma_precision         : 0.852 ± 0.059
                           frame chroma_recall            : 0.753 ± 0.075
                           frame chroma_accuracy          : 0.667 ± 0.081
                           frame chroma_substitution_error: 0.047 ± 0.019
                           frame chroma_miss_error        : 0.200 ± 0.066
                           frame chroma_false_alarm_error : 0.086 ± 0.047
                           frame chroma_total_error       : 0.333 ± 0.091
```