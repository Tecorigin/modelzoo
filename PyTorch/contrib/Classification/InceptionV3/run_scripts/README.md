## 参数介绍：
| 参数名 | 解释 | 样例 |
| - | - | - |
| model_name | 模型名称 |--model_name inceptionv4|
| batch_size/bs | 训练数据的batch_size |--batch_size 128 / --bs 128|
| epoch | 训练轮数 |--epoch 80|
|learning_rate/lr|学习率|--lr 0.1|
|autocast|是否开启AMP训练|--autocast True|
|device|指定训练的设备id|--device sdaa|
|datasets_path|训练数据集文件的路径|--datasets_path ./dataset/cifar10|
|warm|预热学习率的阶段|--warm 1|
|distributed|是否开启DDP|--distributed True|
|nproc_per_node|每个节点上的进程数|--nproc_per_node 4|