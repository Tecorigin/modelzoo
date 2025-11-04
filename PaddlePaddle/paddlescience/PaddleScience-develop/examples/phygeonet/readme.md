 **1.当前软件栈版本** 
![输入图片说明](https://foruda.gitee.com/images/1738900795709577351/8c234fbd_12173785.png "0f53ba650978b265cfb5091f991df07.png")

 **2.源码参考** 
https://github.com/PaddlePaddle/PaddleScience

 **3.工作目录** 
PaddlePaddle\paddlescience\PaddleScience-develop\examples\phygeonet

 **4.数据集获取** 

```
# heat_equation
# linux
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/PhyGeoNet/heat_equation.npz -P ./data/
```

 **5.训练代码** 

```
python heat_equation.py


```
 **6.测试代码** 

```
python heat_equation.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/PhyGeoNet/heat_equation_pretrain.pdparams

```
 **7.结果评测** 
设备成功转移到sdaa
![输入图片说明](https://foruda.gitee.com/images/1738899112998438859/9987f292_12173785.png "0e678ffe81ca8da6fe54f4001f76873.png")
能够长训，且loss有下降趋势
![输入图片说明](https://foruda.gitee.com/images/1739007779224382757/6298b65c_12173785.png "9399d5776fd183036875fbe4bfb6409.png")