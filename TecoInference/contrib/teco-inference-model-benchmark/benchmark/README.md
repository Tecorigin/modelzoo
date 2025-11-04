# benchmark 工具使用说明

## 支持的功能
- 运行指定网络，使用随机输入，获取推理结果的精度，及性能信息
- 精度结果与onnxruntime结果对比，(tvm fp16不准确)
- 精度包括diff2，diff3_max, diff3_min
- 性能由单核组统计，包括h2d， infer， d2h时间

## 参数及使用说明

| 参数名称 | 类型| 必填|  说明 | 备注 |
| --- | :---: | :---: | :---: | :---: |
|--json | str | yes | the path of json file | case的配置信息

e.g python main.py --json ./json/uiex/uie-bs1.json

json文件中包括的配置信息
| 参数名称 | 类型| 必填|  说明 | 备注 |
| --- | :---: | :---: | :---: | :---: |
|model_path  | str  |yes| |  None | onnx模型文件的存放位置|
|dtype  | str  |yes| "float16"|  None | 输入数据及推理使用的数据类型|
|-warm_up    | int  |no| default 5 | warm up times of inferring |前times推理作为warm_up|
|-run_loops     | int  |no| |default 10  |进行10次推理|
|extra_params| dict|no|包含一些额外的参数|如使用真实数据,可加一个字典{'inputs'{'xxx':'path'}},xxx代表输入的名字，path为npy文件路径|

## TLDR
### 网络精度及性能结果输出
```
python main.py --json json/uiex/uie-bs1.json
```

## 应用举例：监控UIE精度
### 准备工作：
1. 下载UIE onnx模型， 拉取test_models 模型文件，模型适配人员需要将模型上传至test_models仓库，仓库管理员会将模型同步更新至/mnt/test_models/下
### 编写json配置文件
```
{
    "model_path": "/mnt/test_models/onnx_models/CV/ocr_cls/cls_1_3_48_192_float16.onnx",
    "dtype": "float16",
    "warm_up": 1,
    "run_loops": 5,
}
```
### 执行指令
```python
python main.py --json json/resnet50/resnet50_bs1_224x224.json
```
### 结果样式
```
10:30:27 resnet50.py[line:127] [INFO]: case path : /mnt/test_models/onnx_models/CV/resnet/resnet_fp16_ngc_bs64.onnx

Test Result:
actual has NaN: False, has Inf: False
desired has NaN: False, has Inf: False
actual shape: (1, 64, 1000)
desired shape: (1, 64, 1000)
diff2 = 0.0014048253108042451
diff3_max = 12.4710693359375, (-0.0015411376953125, 0.00013434886932373047)
diff3_mean = 0.007859268225729465
10:30:27 resnet50.py[line:129] [INFO]: TVM H2D time is: 0.003940582275390625
10:30:27 resnet50.py[line:130] [INFO]: TVM INFER time is: 5.336582660675049
10:30:27 resnet50.py[line:131] [INFO]: TVM D2H time is: 0.00042319297790527344
10:30:27 resnet50.py[line:132] [INFO]: TVM E2E time is: 5.340946435928345
10:30:27 resnet50.py[line:133] [INFO]: ORT E2E time is: 1.521745204925537

```

## 批量测试脚本

### 批量测试 network
``` shell
cd benchmark
# 测试脚本传入json文件的路径
./test_all_network.sh json
```

## 添加测例
参考[编写json配置文件](#编写json配置文件)编写好测例对应json文件后，在benchmark/json目录新建一个模型对应目录如：unet/，并将json配置放入该目录下即可。