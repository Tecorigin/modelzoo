#  参数介绍

| 参数名         | 解释                  | 样例                   |
| -------------- | --------------------- | ---------------------- |
| model_name     | 模型名称              | --model_name mobilenet |
| batch-size     | 训练数据的batch_size  | --batch-size 48        |
| epochs         | 训练轮数              | --epochs 50            |
| lr             | 学习率                | --lr 0.01              |
| autocast       | 是否开启AMP训练       | --autocast True        |
| distributed    | 是否开启DDP训练       | -distributed           |
| nproc_per_node | 每个节点使用的GPU数量 | -nproc_per_node 3      |
| data           | 训练数据集保存地址    | --../data              |
| dataset        | 训练数据集            | --cifar10              |
| seed           | 随机数种子            | --seed 12              |
| num_workers    | 载入数据的线程数      | --num_workers 0        |







