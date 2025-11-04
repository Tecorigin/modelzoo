## yaml格式说明

### 字段含义

- `case_name`: 用户账号name
- `run_dir`: 执行脚本所在目录的相对路径（**路径以TecoInference开头**）
- `run_export`:
  - `run_file`: onnx导出的脚本名
- `run_example_valid`:
  - `run_file`:  执行数据集推理脚本名
  - `params`:
    1. 必要参数如下：
    - `ckpt`: 使用的onnx文件路径
    - `batch_size`: 推理的bs大小
    - `data_path`: 数据集在共享磁盘`/mnt/nvme/common`上的绝对路径
    - `target`: 使用的推理引擎，默认`sdaa`
    - `dtype` : 推理的格式：(`float16|float32`)
    2. 用户自定义参数使用`others`字段来指定
    - `others`:
      - `param1`: val1
      - `params2`: val2
- `run_example_single`:
  - `run_file`:  执行单样本推理脚本名
  - `params`:
    - `ckpt`:
    - `batch_size`:
    - `data_path`: 使用的单样本相对`run_dir`的路径
    - `target`:
    - `dtype` :
    - `others`:
      - `param1`: val1
      - `params2`: val2
- `run_example_multi`:
  - `run_file`:  执行多样本推理推理脚本名
  - `params`:
    - `ckpt`:
    - `batch_size`:
    - `data_path`: 使用的多样本目录相对`run_dir`的路径
    - `target`:
    - `dtype` :
    - `others`:
      - `param1`: val1
      - `params2`: val2

* 注：
    1. 推理使用的onnx文件为导出生成的onnx文件，请确保推理脚本使用的`ckpt`参数与生成的onnx文件路径一致。
    2. **请在`/mnt/nvme/common`以用户名命名一个文件夹，并将使用的数据集放到该文件夹下。如果需要使用其他较大的配置文件/权重文件等，请统一放到上述目录下，并使用对应的参数指定（如仅使用`/mnt/nvme/common`下的数据集，则无需创建文件夹）。**
    2. 必要参数缺一不可。
    3. 所有参数设置不可为空。
    4. 如没有自定义参数，无需添加`others`字段。

### yaml示例
```
- case_name: test_name
  run_dir: TecoInference/example/classification/resnet
  run_export:
    run_file: export_onnx.py
  run_example_valid:
    run_file: example_valid.py
    params:
      ckpt: resnet_float16_dyn.onnx
      batch_size: 64
      data_path: <path>/<to>/imagenet/val
      target: sdaa
      dtype: float16
      others:
        topk: 1
        skip_postprocess: True
        input_size: 224
  run_example_single:
    run_file: example_single_batch.py
    params:
      ckpt: resnet_float16_dyn.onnx
      batch_size: 1
      data_path: images/hen.jpg
      target: sdaa
      dtype: float16
      others:
        topk: 1
        input_size: 224
  run_example_multi:
    run_file: example_multi_batch.py
    params:
      ckpt: resnet_float16_dyn.onnx
      batch_size: 1
      data_path: images
      target: sdaa
      dtype: float16
      others:
        topk: 1
        input_size: 224

```


### 指令执行
pr的测试会根据model.yaml文件来配置，请务必使用正确的yaml格式。以[yaml示例](#yaml示例)为例，测试流程如下：
1. 切换到`TecoInference/example/classification/resnet`目录
2. 执行onnx导出：
    ```sh
    python export_onnx.py
    ```
    在当前目录路径下生成`resnet_float16_dyn.onnx`文件。
3. 执行数据集推理：
    ```sh
    python example_valid.py --ckpt resnet_float16_dyn.onnx --batch_size 64 --data_path <path>/<to>/imagenet/val --target sdaa --dtype float16 --topk 1 --skip_postprocess True --input_size 224
    ```
4. 执行单样本推理：
    ```sh
    python example_single_batch.py --ckpt resnet_float16_dyn.onnx --batch_size 1 --data_path images/hen.jpg --target sdaa --dtype float16 --topk 1 --input_size 224
    ```
5. 执行多样本推理：
    ```sh
    python example_multi_batch.py --ckpt resnet_float16_dyn.onnx --batch_size 1 --data_path images --target sdaa --dtype float16 --topk 1 --input_size 224
    ```
