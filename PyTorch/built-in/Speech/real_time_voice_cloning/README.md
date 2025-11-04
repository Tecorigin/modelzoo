# Real-Time Voice Cloning

## 模型简介：
Real-Time Voice Cloning 是一种基于深度学习的语音克隆模型。它可以将一个人的语音信号克隆到另一个人身上，从而实现语音合成。Real-Time Voice Cloning 模型可以用于语音合成、语音识别、语音翻译等任务。

## 数据集下载：
您可以从 [VCTK](URL_ADDRESS您可以从 [`LibriSpeech/train-clean-100`](https://www.openslr.org/resources/12/train-clean-100.tar.gz) 下载 `train-clean-100` 数据集。解压后，将数据集保存到 `data/LibriSpeech/train-clean-100` 目录下。) 下载 VCTK 数据集。该数据集包含 100 个小时的语音数据，总时长为 100 小时。

## 安装依赖：
pip install -r requirements.txt
安装ffmpeg

## 开始训练：
python train.pypython demo_toolbox.py -d <datasets_root>
