 **1.当前软件栈版本** 
![输入图片说明](https://foruda.gitee.com/images/1738900795709577351/8c234fbd_12173785.png "0f53ba650978b265cfb5091f991df07.png")

 **2.源码参考** 
https://github.com/PaddlePaddle/PaddleScience

 **3.工作目录** 
PaddlePaddle\paddlescience\PaddleScience-develop\examples\lorenz

 **4.数据集获取** 

```
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_training_rk.hdf5 -P ./datasets/
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/lorenz_valid_rk.hdf5 -P ./datasets/
```

 **5.训练代码** 

```
python train_enn.py
python train_transformer.py

```
 **6.测试代码** 

```
python train_enn.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/lorenz/lorenz_pretrained.pdparams
python train_transformer.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/lorenz/lorenz_transformer_pretrained.pdparams EMBEDDING_MODEL_PATH=https://paddle-org.bj.bcebos.com/paddlescience/models/lorenz/lorenz_pretrained.pdparams

```
 **7.结果评测** 
设备成功转移到sdaa
![输入图片说明](https://foruda.gitee.com/images/1738899112998438859/9987f292_12173785.png "0e678ffe81ca8da6fe54f4001f76873.png")
能够长训，且loss有下降趋势
![输入图片说明](https://foruda.gitee.com/images/1738903074506070150/7a4fa0c8_12173785.png "8ca6dc5941868cd8a97a817452e4a5a.png")
![输入图片说明](https://foruda.gitee.com/images/1738903100073731994/0ce6137b_12173785.png "de1a255fbffcadeade2401adcbe067c.png")