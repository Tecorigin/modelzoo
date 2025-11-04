# 参数介绍：

| 参数名              |                                       解释 |                            样例                             |
|:-----------------|-----------------------------------------:|:---------------------------------------------------------:|
| model_name       |                                     模型名称 |                     --model_name clip                     |
| batch_size/bs    |                          训练数据的batch_size |                 --batch_size 16 / --bs 16                 |
| epoch            |                                     训练轮数 |                        --epoch 20                         |
| learning_rate/lr |                                      学习率 |                         --lr 3e-5                         |
| autocast         |                                是否开启AMP训练 |                      --autocast True                      |
| device           |                                    	设备类型 |                       --device sdaa                       |
| datasets_path    |                             训练数据集文件的绝对路劲 |         --datasets_path path/chinese_clip/data/datasets/          |
| checkpoint_path  |                           vit加载预训练权重的绝对路径 | --checkpoint_path path/chinese_clip/data/checkpoints/ViT-B-16-OpenAI.pth |
| bert_ckpt_path   |                         bert加载预训练权重的绝对路径 | --bert_ckpt_path  path/chinese_clip/data/checkpoints/bert-clip-chinese |
| nproc_per_node   | DDP时，每个node上的rank数量。不输入时，默认为1，表示单机单SPA运行 |                    --nproc_per_node 4                     |