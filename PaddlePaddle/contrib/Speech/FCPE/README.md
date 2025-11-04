# FCPE

## 模型概述
FCPE是由ChiTu和ylzz1997两位深度学习语音科学家自行研制的一种基音（f0）预测器，具有不输于crepe模型的精度和高于众多f0预测器的速度。FCPE在众多SVC（Singing Voice Conversion）以及SVS（Singing Voice Synthesis）任务中具有优越的速度与精度表现，因此FCPE已经成为众多语音界模型如 搜为此（So-vits）、地府唱（Diffsinger）、大市唱（AllSing）的f0数据提取模型之一。

## Quick Start Guide

### 1、环境准备

#### 1.1 拉取代码仓

``` bash
git clone https://gitee.com/tecorigin/modelzoo.git
```

#### 1.2 Teco环境准备

``` bash
cd /modelzoo/PaddlePaddle/contrib/Speech/FCPE
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
### 2、数据集与依赖准备
#### 2.1 数据集说明

FCPE支持任何符合格式规范的数据集。
我们强烈推荐你的语音数据集采样率为16k Hz，单声道。使用paddle.audio.load加载数据之后得到的第一个返回值应为一个shape为[1, length]的paddle.Tensor，其中length为音频采样点的数量。推荐语音文件使用.wav格式进行存储。
f0数据使用.npy文件进行存储。numpy.load函数应当能正确加载，且加载之后的ndarray应当具有__len__属性为1的shape。
即npy文件存储的必须是一个1维数组。

推荐数据集：https://opendatalab.com/OpenDataLab/MIR-1K/tree/main
若想要使用该数据集，请注意模型配置与数据集格式对应。


#### 2.3 数据集目录结构

在run_scripts目录下按照该数据结构放置文件：

```
└── data
    ├──train
    │   ├──audio
    │   │   ├──语音1.wav
    │   │   ├──语音2.wav
    │   │   └── ...
    │   └──f0
    │       ├──语音1.npy
    │       ├──语音1.npy
    │       └── ...
    │   
    └──val
       ├──audio
       │   ├──语音1.wav
       │   ├──语音2.wav
       │   └── ...
       └──f0
           ├──语音1.npy
           ├──语音1.npy
           └── ...
```

其中train文件夹下为用于训练的数据，val文件夹下为测试的数据。data文件夹下已经有预先放置好的示例数据集。

#### 2.3 安装依赖

```bash
pip install -r requirements.txt
```

### 3、 启动训练

建议通过编辑run_scripts目录下的config.yaml文件进行参数设置。
``` bash
cd modelzoo/PaddlePaddle/contrib/Speech/FCPE
```
#### 3.1 模型训练脚本参数说明如下：

参数名 | 解释 | 样例 | 是否必须及原因
-----------------|-----------------|-----------------|-----------------
lr | 学习率 | --lr 0.0005 | 否，因为有默认参数
batch_size/bs | 每个rank的batch_size | --batch_size 64 / --bs 64 | 否，因为有默认参数
interval_log | 记录日志的间隔step | --interval_log 1 | 否，因为有默认参数
interval_val | 保存模型的间隔step | --interval_val 500 | 否，因为有默认参数
interval_force_save | 保存不会被清理的模型间隔 | --interval_force_save 1000 | 否，因为有默认参数
epoch | 训练轮次 | --epoch 100000 | 否，因为有默认参数
num_workers | dataloader读取数据进程数 | --num_workers 0 | 否，甚至不能修改，因为paddle会疯狂泄露shm
amp_dtype | 混合精度训练数据类型 | --amp_dtype fp16 | 否，因为有默认参数。默认fp32，即不开混合精度训练
nproc_per_node | DDP时，使用的卡的数量。不输入时，默认为1，跑单卡。| --nproc_per_node 4 | 否，因为有默认参数。默认单卡
model_name |模型名称 | --model_name CFNaiveMelPE | 否，甚至不能修改，因为只有这一种模型

更多参数请参考[更详细的官方文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/launch_cn.html)

#### 3.2 启动，模型

有些算子在sdaa设备上面是不支持的，所以需要设置算子黑名单环境变量：
```bash
# conv2d不支持group非1，因为反向传播还有个单独的算子，所以需要再来个conv2d_grad
export CUSTOM_DEVICE_BLACK_LIST=conv2d,conv2d_grad
```

```bash
# 单卡测试一百个回合
# 这里设置记录日志间隔为1，即每一个step的loss都会被记录
# 所有的log都会被保存在json_log.log里面
python run_scripts/run_script.py --epoch 100 -il 1
```

也可以直接启动train.py
```
# 需要提前编辑config.yaml文件，以进行参数设置
python train.py -c config.yaml
```

或者多卡训练，首先需要配置好相关的环境变量：

见相关的[issue](https://gitee.com/tecorigin/modelzoo/issues/I9S4CZ)
```bash
export PADDLE_XCCL_BACKEND=sdaa
# 注意，这里需要根据你的卡的实际情况设置
export SDAA_VISIBLE_DEVICES=0,1,2,3
```
开始使用DDP多卡训练
```bash
python -m paddle.distributed.launch --devices=0,1,2,3 train.py -c config.yaml
```
Paddlepaddle在版本为2.5.x时只能用这种方式。见相关的[issue](https://gitee.com/tecorigin/modelzoo/issues/I9S1ID)

### 4、推理数据

FCPE已经有了一个训练好的模型，位于FCPE目录下。model.pdparams就是这个训练好的模型了。
可以参考use.py观察如何使用模型预测歌声f0。
