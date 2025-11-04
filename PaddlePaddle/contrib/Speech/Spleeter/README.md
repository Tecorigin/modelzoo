# Spleeter

## 模型概述
Spleeter 是一种用于音频源分离（音乐分离）的开源深度学习算法，由Deezer研究团队开发。在Deezer团队发布的论文《Spleeter: a fast and efficient music source separation tool with pre-trained models》中，他们对Spleeter的总结为一个性能取向的音源分离算法，并且为用户提供了已经预训练好的模型，能够开箱即用，这也是Spleeter爆火的原因之一。其二是Deezer作为一个商业公司，在算法成果发布后迅速与其它产品进行合作，将Spleeter带到了iZotope RX、SpectralLayers、Acoustica、VirtualDJ、NeuralMix等知名专业音频软件中，大大提升了它的知名度。
现在的Spleeter已经被广泛应用于各种SVC（Singing Voice Conversion）以及SVS（Singing Voice Synthesis）任务的数据集来源。众多语音界模型如 搜为此（So-vits）、韵泉唱（YQ maintained diffsinger）、地府唱（Diffsinger）、大市唱（AllSing）都可以使用Spleeter模型的输出作为训练自己的数据集。


## Quick Start Guide

### 1、环境准备

#### 1.1 拉取代码仓

``` bash
git clone https://gitee.com/tecorigin/modelzoo.git
```

#### 1.2 Teco环境准备

``` bash
# 切换到工作目录下
cd modelzoo/PaddlePaddle/contrib/Speech/Spleeter
# 激活paddle的环境
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

### 2、数据集与依赖准备
#### 2.1 数据集说明

Spleeter支持任何符合格式规范的数据集。

推荐数据集：[MUSDB18](https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems "MUSDB18")
推荐数据集：[MUSDB18两轨](https://aistudio.baidu.com/datasetdetail/227900 "MUSDB18两轨")

若想要使用该数据集，请注意模型配置与数据集格式对应。

#### 2.2 数据集目录结构

建议运行以下命令自动放置数据集：

```Bash
python downloadMusdb.py
```

或者在模型目录下按照该数据结构放置文件：

```Text
dataset
├───第一首歌
│      ├───mixture.wav
│      ├───vocal.wav
│      └───instrumental.wav
└───第二首歌
        ├───mixture.wav
        ├───vocal.wav
        └───instrumental.wav
```
在放置好数据之后，<font color=red>必须</font>通过启动preprocess.py对歌曲的信息进行统计：
```Bash
python preprocess.py
```



### 3、 启动训练

这里还需要为训练额外做一点工作。
有些算子在sdaa设备上面是不支持的，所以需要设置算子黑名单环境变量：
```bash
# conv2d不支持group非1，因为反向传播还有个单独的算子，所以需要再来个conv2d_grad
export CUSTOM_DEVICE_BLACK_LIST=conv2d,conv2d_grad
```
设置多卡环境：
```bash
# 请注意！这里请根据你的卡的实际数量进行修改
export SDAA_VISIBLE_DEVICES=0,1,2,3
export PADDLE_XCCL_BACKEND=sdaa
```

首先预加载所有数据。在这个过程当中，将会按照设置对每首歌进行划分并且进行STFT处理。

```Bash
python run_scripts/run_spleeter.py -m process_data
```

之后可以开始训练了：

```bash
python run_scripts/run_spleeter.py -m fast_train -bs 20
```
开始使用DDP多卡训练
```bash
# 请根据你的卡的实际数量进行选择
python run_scripts/run_spleeter.py -m fast_train -npn 4 -bs 20
```

注意：默认是加载预训练模型进行训练的，如想从头训练，请关闭`-lm`参数，如：
```Bash
python run_scripts/run_spleeter.py -m fast_train -npn 4 -bs 20 -lm False
```

### 4、推理数据

Spleeter已经有了一组训练好的模型，位于model目录下。

可参考separator.py中的参数了解如何使用模型分离人声和伴奏。

使用`eval.py`自动加载最新模型并随机抽样十分之一的数据进行推理并计算指标：
```Bash
# 这里请根据你的实际情况填写卡的数量
python eval.py -npn 4 -seed 123456
```
全部推理计算指标完成之后，程序将会在控制台上打印测试指标。

#### 4.1推理效果指标（使用MUSDB18数据集进行测试）

芯片 | 卡 | 模型 | 混合精度 | 训练时的批次大小 |  数据形状  | 平均吞吐量 | SDR | SIR | SAR  | ISR
-----|----|------|----------|----------|------------|--------|-----|-----|------|-----
SDAA | 4  |人声轨|    否    |    20    |[2,音频长度]| 43.370秒/一首歌  | 3.547 | 16.114 | 5.924 | 4.037
SDAA | 4  |伴奏轨|    否    |    20    |[2,音频长度]| 43.370秒/一首歌  | 5.476 |  18.051 | 14.568 | 5.669

其中，在测试的时候，歌曲的时长不会超过6分钟。

测试时音频采样率使用44100Hz，最短的音频长度为1622389，最长的音频长度为15157787，平均音频长度为8901100.733

测试时用时最短为4.142秒，最长用时为88.882秒，平均用时43.370秒。