 **1.当前软件栈版本** 
![输入图片说明](https://foruda.gitee.com/images/1738900795709577351/8c234fbd_12173785.png "0f53ba650978b265cfb5091f991df07.png")

 **2.源码参考** 
https://github.com/PaddlePaddle/PaddleScience

 **3.工作目录** 
PaddlePaddle\paddlescience\PaddleScience-develop\examples\neuraloperator

 **4.数据集获取** 

```
# darcy-flow 数据集下载
# linux
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/neuraloperator/darcy_flow/darcy_train_16.npy -P ./datasets/darcyflow/
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/neuraloperator/darcy_flow/darcy_test_32.npy -P ./datasets/darcyflow/
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/neuraloperator/darcy_flow/darcy_test_16.npy -P ./datasets/darcyflow/
```

 **5.训练代码** 

```
# tfno 模型训练
python train_tfno.py
# uno 模型训练
python train_uno.py

```
 **5.训练代码（cpu）** 

```
# tfno 模型训练
python train_tfno_cpu.py
# uno 模型训练
python train_uno_cpu.py

```
 **6.测试代码** 

```
# tfno 模型评估
python train_tfno.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/neuraloperator/neuraloperator_tfno.pdparams
# uno 模型评估
python train_uno.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/neuraloperator/neuraloperator_uno.pdparams

```
 **7.结果评测** 
sdaa上无法运行，fall back cpu上正常训练
![输入图片说明](https://foruda.gitee.com/images/1738908987753056116/938c24ae_12173785.png "5af734c48cc2f1a9ef48f9c7f4b9153.png")

 **7.不支持算子信息**
otal_loss.backward()
![输入图片说明](https://foruda.gitee.com/images/1739956636285981984/6d743488_12173785.png "屏幕截图")