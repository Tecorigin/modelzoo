 **1.当前软件栈版本** 


 **2.源码参考** 
https://github.com/PaddlePaddle/PaddleScience

 **3.工作目录** 
PaddlePaddle\paddlescience\PaddleScience-develop\examples\deephpms

 **4.数据集获取** 

```
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepHPMs/burgers_sine.mat -P ./datasets/
```

 **5.训练代码** 

```
python burgers.py DATASET_PATH=./datasets/burgers_sine.mat DATASET_PATH_SOL=./datasets/burgers_sine.mat

```
 **6.测试代码** 

```
python burgers.py mode=eval DATASET_PATH=./datasets/burgers_sine.mat DATASET_PATH_SOL=./datasets/burgers_sine.mat EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/DeepHPMs/burgers_same_pretrained.pdparams

```
 **7.结果评测** 
设备成功转移到sdaa
![输入图片说明](https://foruda.gitee.com/images/1738899112998438859/9987f292_12173785.png "0e678ffe81ca8da6fe54f4001f76873.png")
能够长训，且loss有下降趋势
![输入图片说明](https://foruda.gitee.com/images/1738900105979113905/cd263bbd_12173785.png "9d4cbfd99f35310cfb0f5dc8a91d56a.png")