 **1.当前软件栈版本** 


 **2.源码参考** 
https://github.com/PaddlePaddle/PaddleScience

 **3.工作目录** 
PaddlePaddle\paddlescience\PaddleScience-develop\examples\allen_cahn

 **4.数据集获取** 

```
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/AllenCahn/allen_cahn.mat -P ./dataset/
```

 **5.训练代码** 

```
python allen_cahn_piratenet.py
```
 **6.测试代码** 

```
python allen_cahn_piratenet.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/AllenCahn/allen_cahn_piratenet_pretrained.pdparams
```
 **7.结果评测** 
设备成功转移到sdaa
![输入图片说明](https://foruda.gitee.com/images/1738899112998438859/9987f292_12173785.png "0e678ffe81ca8da6fe54f4001f76873.png")
能够长训，且loss有下降趋势
![输入图片说明](https://foruda.gitee.com/images/1738899199602776309/92b52b20_12173785.png "1e30f313257ac7557f8e653f7b130bf.png")
