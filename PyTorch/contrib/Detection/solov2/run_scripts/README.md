以下是补充后的参数介绍：

| 参数名            |                        解释              |                            样例                             |
|:-----------------|-----------------------------------------:|:---------------------------------------------------------:|
| model_name       |                       模型名称            |                     --model_name faster_rcnn                |
| batch_size/bs    |                  训练数据的 batch_size     |                 --batch_size 16 / --bs 16                   |
| epoch            |                             训练轮数      |                        --epoch 10                           |
| lr               |                           学习率          |                         --lr 1e-5                            |
| device           |                       设备类型            |                       --device cuda                          |
| dataset_path         |                   训练数据集文件的路径     |                    --dataset_path data/coco                      |
| nproc_per_node   | DDP 时，每个 node 上的 rank 数量。不输入时，默认为 1，表示单机单 SPA 运行 |         --nproc_per_node 4               |
| nnodes            | 参与分布式训练的节点数目。默认为 1 表示单节点训练 |                       --nnodes 1                             |