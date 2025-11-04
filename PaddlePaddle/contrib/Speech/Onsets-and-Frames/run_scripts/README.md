train.py 参数解释：

参数名 | 解释 
-----------------|-----------------
logdir | VisualDL的日志存放位置
device | 运行脚本时使用的设备，默认为sdaa
nproc_per_node | 使用几张卡进行训练
iterations | 迭代的次数，默认为500000
resume_iteration | 从哪开始训练，默认为None
checkpoint_interval | 模型保存间隔，默认为1000
train_on | 使用的数据集，默认为MAPS，也可以选择MAESTRO
batch_size | 批次大小
sequence_length | 输入音频序列长度
model_complexity | 模型复杂度，默认为48，请勿修改
amp_on | 使用半精度进行训练，不推荐，很容易产生nan
learning_rate | 学习率
learning_rate_decay_steps | 衰减步数
learning_rate_decay_rate | 学习率衰减率
leave_one_out | 额外划分的验证集，可以在'2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017'中任选一个
clip_gradient_norm | 梯度裁剪范数
validation_length | 验证集长度，默认和sequence_length一致。
validation_interval | 验证的间隔step。由于rnn算子不支持推理模型，因此该值应该大于iterations参数。默认为500010