参数介绍
参数名 | 解释 | 样例
-----------------|-----------------|-----------------
--train_file	训练数据文件路径，要求为 .jsonl 格式。	--train_file /data/teco-data/squad/qanything_train.jsonl
--model_name_or_path	本地模型路径，需指向已有的 transformer 模型目录。	--model_name_or_path /data/bigc-data/lsq/QAnything/new_qanything/bge-m3
--output_dir	训练输出目录，用于保存训练模型及日志。	--output_dir ./outputs/emb_m3_sdaa
--log_file	训练过程 loss 的日志保存文件名。	--log_file sdaa_loss.log
--batch_size	每个训练 step 的样本数量。	--batch_size 5
--max_steps	最大训练步数。	--max_steps 100
--lr	学习率，控制优化器参数更新的步幅。	--lr 5e-5
--max_len	分词器允许的最大 token 长度，超出部分会被截断。	--max_len 256
--seed	设置随机种子，保证训练可复现。	--seed 42
--no_amp	关闭自动混合精度（bfloat16），开启该选项则禁用 AMP。	--no_amp（默认不开启）