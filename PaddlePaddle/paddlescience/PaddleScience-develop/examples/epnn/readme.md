 **1.当前软件栈版本** 
![输入图片说明](https://foruda.gitee.com/images/1738900795709577351/8c234fbd_12173785.png "0f53ba650978b265cfb5091f991df07.png")

 **2.源码参考** 
https://github.com/PaddlePaddle/PaddleScience

 **3.工作目录** 
PaddlePaddle\paddlescience\PaddleScience-develop\examples\epnn

 **4.数据集获取** 

```
# linux
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstate-16-plas.dat -P ./datasets/
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/epnn/dstress-16-plas.dat -P ./datasets/
```

 **5.训练代码** 

```
python epnn.py

```
 **6.测试代码** 

```
python epnn.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/epnn/epnn_pretrained.pdparams

```
 **7.结果评测** 
设备成功转移到sdaa
![输入图片说明](https://foruda.gitee.com/images/1738899112998438859/9987f292_12173785.png "0e678ffe81ca8da6fe54f4001f76873.png")
能够长训，且loss有下降趋势
![输入图片说明](https://foruda.gitee.com/images/1739005565568387450/a268a41c_12173785.png "7283550f5239ecf5a7fd5bddc51f141.png")