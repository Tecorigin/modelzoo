 **1.当前软件栈版本** 
![输入图片说明](https://foruda.gitee.com/images/1738900795709577351/8c234fbd_12173785.png "0f53ba650978b265cfb5091f991df07.png")

 **2.源码参考** 
https://github.com/PaddlePaddle/PaddleScience

 **3.工作目录** 
PaddlePaddle\paddlescience\PaddleScience-develop\examples\ldc

 **4.数据集获取** 

```
wget -nc -P ./data/ \
    https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re100.mat \
    https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re400.mat \
    https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re1000.mat \
    https://paddle-org.bj.bcebos.com/paddlescience/datasets/ldc/ldc_Re3200.mat
```

 **5.训练代码** 

```
python ldc_2d_Re3200_sota.py

```
 **6.测试代码** 

```
python ldc_2d_Re3200_sota.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/ldc/ldc_re1000_sota_pretrained.pdparams

```
 **7.结果评测** 
设备成功转移到sdaa
![输入图片说明](https://foruda.gitee.com/images/1738899112998438859/9987f292_12173785.png "0e678ffe81ca8da6fe54f4001f76873.png")
能够长训，且loss有下降趋势
