**1.当前软件栈版本** 
![输入图片说明](https://foruda.gitee.com/images/1738900795709577351/8c234fbd_12173785.png "0f53ba650978b265cfb5091f991df07.png")

 **2.源码参考** 
https://github.com/PaddlePaddle/PaddleScience

 **3.工作目录** 
PaddlePaddle\paddlescience\PaddleScience-develop\examples\cylinder/2d_unsteady\transformer_physx

 **4.数据集获取** 

```
# linux
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/cylinder_training.hdf5 -P ./datasets/
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/transformer_physx/cylinder_valid.hdf5 -P ./datasets/
```

 **5.训练代码** 

```
python train_enn.py
python train_transformer.py

```
 **5.训练代码（cpu）** 

```
python train_enn_cpu.py
python train_transformer_cpu.py

```
 **6.测试代码** 

```
python train_enn.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/cylinder/cylinder_pretrained.pdparams
python train_transformer.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/cylinder/cylinder_transformer_pretrained.pdparams EMBEDDING_MODEL_PATH=https://paddle-org.bj.bcebos.com/paddlescience/models/cylinder/cylinder_pretrained.pdparams

```
 **7.结果评测** 
sdaa算子不支持，fall back cpu上可以正常训练，错误如下：
![输入图片说明](https://foruda.gitee.com/images/1738926483054662757/e29c7c32_12173785.png "bfbc0f122e11fc94a196bb3b4af3e00.png")
算子错误信息：
 File "/root/miniconda3/envs/paddle_env/lib/python3.10/site-packages/ppsci/arch/embedding_koopman.py", line 494, in get_koopman_matrix
    kMatrix[self.xidx, self.yidx] = kMatrixUT_data_t
![输入图片说明](https://foruda.gitee.com/images/1739957441148782508/43f35f15_12173785.png "屏幕截图")