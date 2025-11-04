# WaveGAN

## 模型简介：
WaveGAN 是一种基于生成对抗网络 (GAN) 的音频生成模型。它使用了一种称为 WaveNet 的架构，能够生成高保真度的音频信号。WaveGAN 模型可以用于音频合成、音频去噪、音频增强等任务。

## 数据集下载：
您可以从 [LJSpeech](URL_ADDRESS您可以从 [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) 下载 LJSpeech 数据集。该数据集包含 13,100 条英文语音数据，总时长为 24 小时。) 下载 LJSpeech 数据集。该数据集包含 13,100 条英文语音数据，总时长为 24 小时。

## 安装依赖：
pip install -r requirements.txt

## 开始训练：
python WaveGAN_train.py
