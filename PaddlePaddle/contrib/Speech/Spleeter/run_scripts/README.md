# 训练参数介绍
通过执行`run_scripts/run_spleeter.py`来启动训练
训练脚本的参数决定了要启动的功能。所有的参数都是默认的。

参数 | 解释 | 例子
-----------------|-----------------|-----------------
mode | 要执行的功能 | --mode train
nproc_per_node | 调用多少张卡，在mode为process_data时只能填1。默认值为1 | --nproc_per_node 4
margin             | 电平余量                                       | --margin 0.5
chunk_duration     | 区块持续时间 								    |   --chunk_duration 20.0
sample_rate        | 采样率 										|  --sample_rate 44100
frame_length       | 提取频谱时的帧长 						    	| --frame_length 4096   
frame_step         | 提取频谱时的帧移							    | --frame_step 1024  
T                  | 特征输入时分块长度 							|--T 512     
F                  | 特征频谱上限 							    	|   --F 1024  
n_chunks_per_song  | 每首歌的分块个数  						    	| --n_chunks_per_song 15
train_manifest     | 训练集清单，由preprocess.py生成				| --train_manifest manifest.csv
train_dataset      | 数据文件夹  									|--train_dataset dataset
epochs             | 训练最大回合  								    |   --epochs 10       
batch_size         | 每一张卡上的批次大小  		        	    	| --batch_size 20
optimizer          | 使用的优化器  									|  --optimizer 'adam' 
loss               | 损失函数  								    	| --loss 'l1'       
momentum           | 优化器动量  									|--momentum .9        
dampening          | 抑制									    	|   --dampening 0.  
lr                 | 学习率 								    	|  --lr 1e-5         
lr_decay           | 学习率衰减率							     	|   --lr_decay 0. 
wd                 | 权值衰减								        | --wd 0.00001
model_dir          | 训练模型保存路径 						     	| --model_dir model   
load_optimizer     | 加载上次训练的优化器 						    |   --load_optimizer True
start              | 起始训练的轮数 								|  --start 0
load_model         | 是否恢复训练（或者使用预训练模型）			    |   --load_model True
seed               | 随机种子    									| --seed 37      
keep_ckpt          | 保留模型数量 								    |   --keep_ckpt 1
trainer            | 训练者名称，可以在这里写上你的名字 		    |--trainer daShiChangZhaoLei
clean_logs         | 及时清理控制台信息  							|--clean_logs True      
amp                | 是否开启混合精度训练，不推荐开启，会损失精度	|   --amp False         

其中对mode参数的介绍如下：

模式 | 解释 
-----------------|-----------------
train | 一边加载数据一边训练，但是这种训练方式容易被IO瓶颈影响性能。
process_data | 将所有数据先保存为张量，到训练的时候直接进行加载，几乎没有IO瓶颈，可以大大节省时间。但是会固定部分参数，包括n_chunks_per_song，但是不会固定batch size。
fast_train | 在使用process_data处理完所有数据之后，使用已经处理完成的数据进行训练。可以极大提高训练效率。
