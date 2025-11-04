**1.当前软件栈版本** 
![输入图片说明](https://foruda.gitee.com/images/1738900795709577351/8c234fbd_12173785.png "0f53ba650978b265cfb5091f991df07.png")

 **2.源码参考** 
https://github.com/PaddlePaddle/PaddleScience

 **3.工作目录** 
PaddlePaddle\paddlescience\PaddleScience-develop\examples\hpinns

 **4.数据集获取** 

```
# linux
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_train.mat -P ./datasets/
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/hPINNs/hpinns_holo_valid.mat -P ./datasets/
```

 **5.训练代码** 

```
python holography.py

```
 **6.测试代码** 

```
python holography.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/hPINNs/hpinns_pretrained.pdparams

```
 **7.结果评测** 
设备成功转移到sdaa
![输入图片说明](https://foruda.gitee.com/images/1738899112998438859/9987f292_12173785.png "0e678ffe81ca8da6fe54f4001f76873.png")
能够长训，且loss有下降趋势
![输入图片说明](https://foruda.gitee.com/images/1739010013021909330/033a02f4_12173785.png "9752a5ca9501b3210ee73474e3d7bd3.png")