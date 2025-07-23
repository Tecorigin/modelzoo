参数名 | 说明 | 示例
-----------------|-----------------|-----------------
batch_size | 训练批次大小 | --batch_size 128
num_workers | 数据加载的线程数 | --num_workers 4
lr | 模型学习率 | --lr 0.1
weight_decay | 权重衰减 | --weight_decay 1e-4
momentum|动量 | --momentum 0.9
cuda | 是否使用sdaa加速(原代码指cuda加速。这里迁移到sdaa卡，实际表示用sdaa加速) | --cuda True
epochs | 训练轮次 | --epochs 310
print_intervals | 打印日志的间隔步数 | --print_intervals 100
evaluation | 是否进行评估模式 | --evaluation False
checkpoints | 模型检查点保存路径 | --checkpoints /path/to/checkpoints
device_num | 使用的设备数量 | --device_num 1
gradient_clip | 梯度裁剪阈值 | --gradient_clip 2.0