## 参数介绍

参数名 | 说明 | 示例
-----------------|-----------------|-----------------
model_name |模型名称。 | --model_name resnet50
epoch| 训练轮次，和训练步数冲突。 | --epoch 50
step | 训练步数，和训练轮数冲突。 | --step 1
batch_size/bs | 每个rank的batch_size。 | --batch_size 32 / --bs 32
dataset_path | 数据集路径。 | --dataset_path
nproc_per_node | DDP时，每个node上的rank数量。不输入时，默认为1，表示单SPA运行。 | --nproc_per_node 4
nnode | DDP时，node数量。不输入时，默认为1，表示单机运行。| --nnode 2
node_rank|多机时，node的序号。|--node_rank 0
master_addr|多机时，主节点的IP地址。|--master_addr 127.0.0.1
master_port|多机时，主节点的端口号。|--master_port 13400
precision_align|开启逐层精度对齐。|--precision_align True
precision_align_cuda_path|指定逐层精度对齐cuda-golden基准数据。|--precision_align_cuda_path compare_results_all_50_pt
precision_align_log_path|指定逐层精度对齐日志输出结果。|--precision_align_log_path precision_align_log/50_pt
lr|学习率。|--lr 1e-3
num_workers|dataloader读取数据进程数。|--num_workers 2
device|设备类型。|--device sdaa
profiler|是否开启性能分析。|--profiler True
autocast|是否开启AMP训练。|--autocast True
grad_scale| 是否开启梯度缩放。 | --grad_scale True
compile| 使用compile编译优化训练性能 | --compile True

## 参数设置
*  `model_name` 目前可选参数为 `fused_resnet50`, `resnet50`, `resnet50_v10`，`resnet18`, `resnet101`。模型介绍详见[model introduction](../docs/mdoel_introduction.md)