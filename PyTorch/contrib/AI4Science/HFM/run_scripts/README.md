## 参数介绍
  
|参数名|作用|选项|
|------|----|---|
|path|数据集路径|自定义|
|batch_size|batchsize大小|自定义|
|T_data|数据集时序长度|根据数据集大小严格填写|
|N_data|数据集空间点数量|根据数据集点的个数严格填写|
|work_size|使用的GPU数量|默认是2，根据gpu数量自定义选择|
|total_epoch|总的训练周期|自定义|
|is_eqns|开启自生成空间点功能的标志|True/False|
|is_other_loss|开启辅助loss功能的标志|True/False|
|model_path|预训练模型路径|自定义|
|alpha|开启辅助loss时原始loss所占权重|自定义range(0,1)|
|rey|rey系数|自定义|
|pec|pec系数|自定义|
|layers|神经网络层数，默认10层|自定义|  
|width|网络宽度，默认200|自定义|
|activation|指定模型中使用的激活函数|swish, relu, tanh 等,默认值: 'swish'|
|normalization|指定模型中使用的归一化方法|默认值 None。例如，batchnorm, layernorm 等|
|model_save_path|模型权重保存的路径和文件名|"weights/Cylinder3D.pth"|
|multi_machine|表示启用多机分布式训练模式| |
|master_addr|分布式训练中的主节点地址|"127.0.0.1"|
|master_port|分布式训练中主节点的端口号|'29500'|
|node_rank|当前节点的排名（或编号）|0|
|local_size|每个节点上的本地进程数量|2|
|device|指定训练使用的设备|可选值包括 cpu, cuda, sdaa。默认值为 cpu。|
|ddp | 是否开启分布式 | --ddp|


