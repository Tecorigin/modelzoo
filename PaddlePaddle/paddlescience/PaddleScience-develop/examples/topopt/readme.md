 **1.当前软件栈版本** 
![输入图片说明](https://foruda.gitee.com/images/1738900795709577351/8c234fbd_12173785.png "0f53ba650978b265cfb5091f991df07.png")

 **2.源码参考** 
https://github.com/PaddlePaddle/PaddleScience

 **3.工作目录** 
PaddlePaddle\paddlescience\PaddleScience-develop\examples\operator_learning

 **4.数据集获取** 

```
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/topopt/top_dataset.h5 -P ./datasets/
```

 **5.训练代码** 

```
python topopt.py

```
 **6.测试代码** 

```
python topopt.py mode=eval 'EVAL.pretrained_model_path_dict={'Uniform': 'https://paddle-org.bj.bcebos.com/paddlescience/models/topopt/uniform_pretrained.pdparams', 'Poisson5': 'https://paddle-org.bj.bcebos.com/paddlescience/models/topopt/poisson5_pretrained.pdparams', 'Poisson10': 'https://paddle-org.bj.bcebos.com/paddlescience/models/topopt/poisson10_pretrained.pdparams', 'Poisson30': 'https://paddle-org.bj.bcebos.com/paddlescience/models/topopt/poisson30_pretrained.pdparams'}'

```
 **7.结果评测** 
设备成功转移到sdaa
![输入图片说明](https://foruda.gitee.com/images/1738899112998438859/9987f292_12173785.png "0e678ffe81ca8da6fe54f4001f76873.png")
能够长训，且loss有下降趋势
![输入图片说明](https://foruda.gitee.com/images/1739006399294717714/c40c648d_12173785.png "e89aaa037ff332e0b7201afecc3842f.png")