## 参数介绍

参数名 | 说明 | 示例
-------|------|------
name | 训练任务的名称，自动包含时间戳 | --name imagenet_`date +%F_%H%M%S`
model | 使用的模型架构 | --model BiT-M-R50x1
logdir | 日志文件保存路径 | --logdir /tmp/bit_logs
dataset | 数据集名称 | --dataset imagenet2012
datadir | 数据集存储路径 | --datadir /data/teco-data/imagenet
batch | 训练批次大小 | --batch 4
base_lr | 基础学习率 | --base_lr 0.00025
workers | 数据加载的工作进程数 | --workers 8
eval_every | 每多少步进行一次评估 | --eval_every 100
batch_split | 批次分割数 | --batch_split 4