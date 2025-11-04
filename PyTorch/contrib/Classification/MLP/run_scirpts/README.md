## 参数介绍

| 参数名         | 解释                                                         | 样例                    |
| -------------- | ------------------------------------------------------------ | ----------------------- |
| epochs         | 训练轮次                                                     | --epochs 10             |
| batch_size     | 每个rank的batch_size                                         | --batch_size 64         |
| lr             | 学习率                                                       | --lr  0.001             |
| device         | 设备类型                                                     | --device sdaa           |
| nnode          | DDP时，node数量。不输入时，默认为1，表示单机运行             | --nnode 2               |
| node_rank      | 多机时，node的序号                                           | --node_rank 0           |
| master_addr    | 多机时，主节点的IP地址                                       | --master_addr 127.0.0.1 |
| master_port    | 多机时，主节点的端口号                                       | --matser_port 13400     |
| nproc_per_node | DDP时，每个node上的rank数量。不输入时，默认为1，表示单SPA运行。 | --nproc_per_node 4      |

