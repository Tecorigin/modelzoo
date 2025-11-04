# save_engine 工具使用说明
## 支持的功能
- 将指定的onnx模型，序列化成 tecoInference 引擎

设置参数介绍
| 参数名称 |参数名称| 必填|  说明 | 备注 |
| --- | :---: | :---: | :---: | :---: |
|onnx_path  |-op  |yes|模型路径 | 将 onnx 模型路径。|
|pass_path  |-pp  |yes|最优pass路径| 最优pass路径|
|save_path  |-sp  |no |tecoInference 引擎保存路径|引擎保存路径，默认./xxx.onnx.tecoengine|
|onnx_dtype |     |no |onnx 模型类型  | 默认 float16|
## 使用示例
```
python save_engine.py -op /path/to/resnet_fp16_ngc_bs64.onnx -pp /path/to/onnx_models/CV/resnet/resnet_pass.py
```

# tecoexec 工具使用说明
## 支持的功能
- 运行save_engine工具序列化后的 tecoInference 引擎，根据设置参数，使用随机输入，获取性能数据（不保证精度）

设置参数介绍
| 参数名称 | 必填|  说明 | 备注 |
| --- | :---: | :---: | :---: |
|loadEngine  |yes|None | 加载序列化的 tecoInference 引擎。|
|fp16  |no| "float16"| 输入数据及推理使用的数据类型|
|input_shapes  |yes| 设置网络执行的输入 |1 * 3 * 224 * 224 的形式作为输入，可以通过逗号分隔值提供多个输入形状。|
|iterations  |no| default 50  | 进行50次推理 |
|warmUp  |no| default 10 | 前N次推理作为warm_up |
|runSync  |no| default disable | 启用运行同步 |
|input_dtype  |no| default disable | 设置输入数据类型，通过逗号分隔值提供多个类型，须与input_shapes数量一致且对应。支持类型: bool, fp16, fp32, int32, int64(默认类型为fp16) |

## 使用示例
## resnet网络执行实例
```
 ./tecoexec --loadEngine=/path/to/resnet_64.tecoengine --input_shapes=256*3*224*224 --fp16 --warmUp=10 --iterations=50
```
## uie-x-base网络执行实例
```
 ./tecoexec --loadEngine=/path/to/new_uiex_fp16_0908_bs1.onnx.tecoengine --input_shapes=4*512,4*512,4*512,4*512,4*512*4,4*3*224*224 --fp16 --warmUp=10 --iterations=50 --input_dtype=int64,int64,int64,int64,int64,fp16
```
