### 参数介绍

参数名 | 解释 | 样例
-----------------|-----------------|-----------------
model_name |模型名称。 | --model_name mrt
epoch | 训练轮数。 | --epoch 20
batch_size/bs | batch_size。 |  --batch_size 64 / --bs 64
device | 用于训练或推理的设备。 | --device sdaa / --device_id cpu
data_path | 数据集目录位置。 | --data_path ./mocap
eval | 是否进行验证。 | --eval
autocast | 是否开启AMP训练。 | --autocast
ddp | 是否开启ddp训练策略。 | --ddp
checkpoint | 验证时的权重路径。 | --checkpoint saved_model/19.model
