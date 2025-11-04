## 示例命令

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 train_cosface.py \
    --database WebFace \
    --batch_size 512 \
    --epochs 30 \
    --lr 0.01 \
    --network sphere20 \
    --classifier_type MCP \
    --save_path /data/ckpt/CosFace_pretrain \
    --workers 4 \
    2>&1 | tee train_sdaa_3rd.log
```


## 参数说明

下表列出了脚本中可用的主要参数，包含参数名称、描述、默认值以及示例用法。可根据自身需求进行修改。

| 参数                    | 描述                                                                                                               | 默认值                           | 示例                                                         |
|-------------------------|--------------------------------------------------------------------------------------------------------------------|----------------------------------|--------------------------------------------------------------|
| **--root_path**         | 训练数据集的根目录，如果数据列表（`train_list`）中的图像路径是相对路径，则需与该根目录拼接。                        | `''`（空字符串）                | `--root_path /data/datasets`                                 |
| **--database**          | 指定使用哪个数据集的默认配置（目前支持 `WebFace`、`VggFace2`），会根据该值设置 `train_list`、`num_class`、`step_size`。 | `WebFace`                        | `--database VggFace2`                                        |
| **--train_list**        | 训练数据的图像列表路径（文件中每行包含 图像相对/绝对路径 + 标签）。若不指定，脚本会根据 `database` 自动设置。          | `None`（根据数据库自动设置）      | `--train_list /path/to/train_list.txt`                       |
| **--batch_size**        | 训练时每个 GPU/进程上的 batch size。                                                                              | `512`                            | `--batch_size 256`                                           |
| **--is_gray**           | 是否将输入图像转换为灰度图（如部分网络结构或数据集需要）。                                                         | `False`                          | `--is_gray True`                                             |
| **--network**           | 使用的网络结构名称，可选：`sphere20`、`sphere64`、`LResNet50E_IR`。                                               | `sphere20`                       | `--network LResNet50E_IR`                                    |
| **--num_class**         | 类别（ID）总数，如果不指定，则根据 `database` 自动设置。                                                           | `None`（根据数据库自动设置）      | `--num_class 10572`                                          |
| **--classifier_type**   | 分类器的类型，可选：`MCP` (MarginCosineProduct)、`AL` (AngleLinear)、`L` (普通线性层)。                             | `MCP`                            | `--classifier_type AL`                                       |
| **--epochs**            | 训练的总轮数（epoch 数）。                                                                                         | `30`                             | `--epochs 50`                                                |
| **--lr**                | 学习率（learning rate）。                                                                                          | `0.01`                           | `--lr 0.001`                                                 |
| **--step_size**         | 学习率衰减的 step（迭代数级别），若不指定，会根据默认数据库设置自动赋值。                                          | `None`（根据数据库自动设置）      | `--step_size 16000 24000`（空格分隔传入列表）                |
| **--momentum**          | SGD 的动量（momentum）。                                                                                           | `0.9`                            | `--momentum 0.95`                                            |
| **--weight_decay**      | weight decay（L2 正则化系数）。                                                                                   | `5e-4`                           | `--weight_decay 1e-4`                                        |
| **--log_interval**      | 日志打印间隔（多少个 batch 打印一次训练信息）。                                                                    | `100`                            | `--log_interval 200`                                         |
| **--save_path**         | 用于保存检查点（checkpoint）的目录。                                                                               | `/data/ckpt/CosFace_pretrain`    | `--save_path /path/to/output`                                |
| **--workers**           | DataLoader 的 worker 线程数。                                                                                      | `4`                              | `--workers 8`                                                |
| **--rank**              | 当前进程在所有进程中的 rank，分布式训练使用；由 `torchrun` 或环境变量自动设置。                                    | `0`                              | 一般无需手动设置                                             |
| **--world_size**        | 分布式训练的所有节点进程数之和，分布式训练使用；由 `torchrun` 或环境变量自动设置。                                 | `1`                              | `--world_size 8`（当总进程=8 时）                            |
| **--gpu**               | 当前进程所使用的本地 GPU id，分布式训练使用；由 `torchrun` 或环境变量 `LOCAL_RANK` 自动设置。                       | `0`                              | 一般无需手动设置                                             |

**说明：**  
- `--database` 设置为 `WebFace` 或 `VggFace2` 时，脚本会自动对 `train_list`、`num_class`、`step_size` 等进行默认配置，若你手动指定了这些参数，会覆盖默认值。
---

## 分布式训练说明

- 使用 `torchrun --standalone --nnodes=1 --nproc_per_node=4`：
  - `--standalone` 表示无需联系其它节点的主服务器进行协调。  
  - `--nnodes=1` 表示只有一个节点（一台机器）。  
  - `--nproc_per_node=4` 表示在该节点上启动 4 个进程（通常对应 4 张 GPU）。  
- 若需要多节点或更多 GPU，请调整相应的参数（如 `--nnodes` 或 `--nproc_per_node` 以及 `--world_size`）。

---

## 日志与检查点

- 检查点会自动保存到 `--save_path` 指定的文件夹下，保存为一个以 `CosFace_{epoch}_checkpoint.pth` 命名的模型文件。   
- 示例命令中的 `2>&1 | tee train_sdaa_3rd.log` 用于将所有终端输出（包括错误信息）保存到 `train_sdaa_3rd.log` 文件中，便于之后查看。

