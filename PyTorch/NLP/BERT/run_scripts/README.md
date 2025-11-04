### 参数介绍

参数名 | 解释 | 样例
-----------------|-----------------|-----------------
model_name |模型名称。 | --model_name bert_base_uncased
epoch | 训练轮数，和训练轮数冲突。 | --epoch 10
step | 训练步数，和训练轮数冲突。 | --step 10
batch_size/bs | 每个rank的batch_size。 | --batch_size 4 / --bs 4
dataset_path | 数据集路径。 | --dataset_path path/to/dataset
nproc_per_node | DDP时，每个node上的rank数量。不输入时，默认为1，表示单机单SPA运行。 | --nproc_per_node 4
lr|学习率|--lr 3e-5
device|设备类型。|--device cuda/--device sdaa
autocast|是否开启AMP训练。|--autocast True
grad_scale| 是否开启梯度缩放。 | --grad_scale True
checkpoint_path| 预训练权重路径。 | --checkpoint_path path/to/bert_base.pt
warm_up| warm_up比例。 | --warm_up 0.2
eval_step | 是否进行验证。 | --eval_step 1
eval_step | 验证的step数。|--eval_step 1000
max_seq_length|输入的最大句子长度。| --max_seq_length 384
precision_align|开启逐层精度对齐。|--precision_align True
precision_align_cuda_path|指定逐层精度对齐cuda-golden基准数据。|--precision_align_cuda_path  ./compare_results_all
precision_align_log_path|指定逐层精度对齐日志输出结果。|--precision_align_log_path ./compare_results_all/gpt2
profiler|是否开启profiler。|--profiler True
profiler_path| Profiler结果保存位置。 | --profiler_path ./json_sdaa
