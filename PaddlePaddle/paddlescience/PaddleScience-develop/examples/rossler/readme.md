 **1.当前软件栈版本** 
![输入图片说明](https://foruda.gitee.com/images/1738900795709577351/8c234fbd_12173785.png "0f53ba650978b265cfb5091f991df07.png")

 **2.源码参考** 
https://github.com/PaddlePaddle/PaddleScience

 **3.工作目录** 
PaddlePaddle\paddlescience\PaddleScience-develop\examples\rossler

 **4.数据集获取** 

```
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_training.hdf5 -P ./datasets/
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/rossler_valid.hdf5 -P ./datasets/
```

 **5.训练代码** 

```
python train_enn.py
python train_transformer.py

```
 **6.测试代码** 

```
python train_enn.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/rossler/rossler_pretrained.pdparams
python train_transformer.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/rossler/rossler_transformer_pretrained.pdparams EMBEDDING_MODEL_PATH=https://paddle-org.bj.bcebos.com/paddlescience/models/rossler/rossler_pretrained.pdparams

```
 **7.结果评测** 
设备成功转移到sdaa
![输入图片说明](https://foruda.gitee.com/images/1738899112998438859/9987f292_12173785.png "0e678ffe81ca8da6fe54f4001f76873.png")
能够长训，且loss有下降趋势
![输入图片说明](https://foruda.gitee.com/images/1738906772990082508/59286aae_12173785.png "53e9e8427bb2939faa25a341771a3a6.png")
![输入图片说明](https://foruda.gitee.com/images/1738906792559729724/ecbe88ce_12173785.png "e1944fb2ab78ee9795e6499bf69a29e.png")