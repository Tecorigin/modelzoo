以下是补充后的参数介绍：

| 参数名            |                        解释              |                            样例                             |
|:-----------------|-----------------------------------------:|:---------------------------------------------------------:|
| model_name       |                       模型名称            |                     --model_name faster_rcnn                |
| batch_size/bs    |                  训练数据的 batch_size     |                 --batch_size 16 / --bs 16                   |
| epoch            |                             训练轮数      |                        --epoch 10                           |
| lr               |                           学习率          |                         --lr 1e-5                            |
| device           |                       设备类型            |                       --device cuda                          |
| data-dir         |                   训练数据集文件的路径     |                    --data-dir data/coco                      |
| nproc_per_node   | DDP 时，每个 node 上的 rank 数量。不输入时，默认为 1，表示单机单 SPA 运行 |         --nproc_per_node 4               |
| nnode            | 参与分布式训练的节点数目。默认为 1 表示单节点训练 |                       --nnode 1                             |
| node_rank        | 当前节点在多节点中的排名。从 0 开始编号 |                    --node_rank 0                            |