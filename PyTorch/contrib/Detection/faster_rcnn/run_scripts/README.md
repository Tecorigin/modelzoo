### 参数介绍

参数名 | 解释 | 样例
-----------------|-----------------|-----------------
model_name |模型名称。 | --model_name faster_rcnn
epoch | 训练轮数，和训练轮数冲突。 | --epoch 10
batch_size | 每个rank的batch_size。 | --batch_size 4
nproc_per_node | DDP时，每个node上的rank数量。不输入时，默认为1，表示单机单SPA运行。 | --nproc_per_node 4
lr|学习率|--lr 3e-5
device|设备类型。|--device sdaa
use_amp|是否开启AMP训练。|--use_amp True
use_ddp|是否开启AMP训练。|--use_ddp True

