# tecoexec
## 简介
随机初始化构造输入，获取模型特定batchsize+shape下极限性能

## 环境准备

请参考[环境准备](../README.md)安装基础环境，若是Host环境，请按如下准备tecoexec性能测试环境：

```
cd modelzoo/TecoInference/tecoexec

pip install -r requirements.txt
wget http://jfrog.tecorigin.net/artifactory/tecobs-ap-teco_inference/artifacts/1.2.0/release/1.2.0/149/tecoexec && chmod +x tecoexec

export LD_LIBRARY_PATH=/<path_to_anaconda>/anaconda3/envs/<your_conda_env>/lib/python3.8/site-packages/tvm/:$LD_LIBRARY_PATH
export TECOEXEC_PATH=/<path_to_modelzoo>/modelzoo/TecoInference/tecoexec/
```

注：请替换`<path_to_anaconda>`, `<your_conda_env>`和`<path_to_modelzoo>`


## 操作指南
```
# Docker
cd /workspace/TecoInference/tecoexec/

# Host
cd modelzoo/TecoInference/tecoexec

pytest -vs -k 'modelname' test_tecoexec.py
```
其中 `modelname`，可替换为：resnet50

注：1. [测例配置](./testcase_configs/tecoexec_config.yaml)；2. 输出日志：test_tecoexec_logs/日期/

## 性能数据
### ResNet50
|shape|latency|throughput|
|:-:|:-:|:-:|
|1x224x224| 3.69 ms| 277.46|
|4x224x224| 3.81 ms| 1075.76|
|8x224x224| 4.54 ms| 1812.05|
|16x224x224| 5.91 ms| 2801.61|
|32x224x224| 5.57 ms| 6147.57|
|64x224x224| 9.2 ms| 7549.44|
|128x224x224| 16.4 ms| 8403.69|
|256x224x224| 32.17 ms| 8993.31|
|512x224x224| 74.98 ms| 7565.88|
