## yaml格式说明

### 字段含义

- `case_name`: 用户账号name
- `desc`:
    - `run_dir`: 执行脚本所在目录的相对路径（**路径以PyTorch或者PaddlePaddle开头即可**）
    - `run_file`: 执行的脚本（**以run_scripts开头**）
- `params`:
    1. 必要参数如下：
    - `nnodes`: DDP下，节点（机器）数量，默认为1
    - `nproc_per_node:` DDP下，每个节点（机器）的显卡数量
    - `batch_size`: 训练时的bs大小
    - `model_name`: 模型名
    - `dataset_path`: 数据集在**共享磁盘`/mnt/nvme/common`上的绝对路径**。
    - `step`: 短训的step数量
    - `epoch`: 长训的epoch数量
    - `lr`: 学习率大小
    - `device`: 使用的计算设备，默认请设置为`sdaa`
    - `autocast`: 是否开启混合精度训练（True/False）

    2. 自定义参数，使用`others`字段来指定：
    - `others`:
      - `param1`: val1
      - `params2`: val2
* 注：
    1. **请在`/mnt/nvme/common`以用户名命名一个文件夹，并将数据集放到该文件夹下。如果需要使用其他较大的配置文件/权重文件等，请统一放到上述目录下，并使用对应的参数指定。**
    2. 必要参数缺一不可。
    3. 参数设置不可为空。
    4. `step`以及`epoch`无需设置较大数值。
    5. 没有自定义参数无需添加`others`字段。

### yaml示例
```
- case_name: test_name
  desc:
    run_dir: PyTorch/NLP/BERT
    run_file: run_scripts/run_bert_base_imdb.py
  params:
    nnodes: 1
    nproc_per_node: 1
    batch_size: 4
    model_name: bert
    dataset_path: <path>/<to>/dataset
    step: 10
    epoch: 1
    lr: 1
    device: sdaa
    autocast: True
    others:
      max_seq_length: 128
      warm_up: 0.1
      checkpoint_path: <path>/<to>/bert_base.pt
```


### 指令执行
pr的测试会根据model.yaml文件来配置，请务必使用正确的yaml格式。以[yaml示例](#yaml示例)为例，测试流程如下：
1. 切换到`PyTorch/NLP/BERT`目录
2. 执行指令：
    ```sh
    python run_scripts/run_bert_base_imdb.py --nnodes 1  --nproc_per_node 1 --batch_size 4 --model_name bert --dataset_path <path>/<to>/dataset --step 10 --epoch 1 --lr 1 --device sdaa --max_seq_length 128 --warm_up 0.1 --checkpoint_path <path>/<to>/bert_base.pt
    ```
