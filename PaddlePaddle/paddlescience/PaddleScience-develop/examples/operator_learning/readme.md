 **1.当前软件栈版本** 
![输入图片说明](https://foruda.gitee.com/images/1738900795709577351/8c234fbd_12173785.png "0f53ba650978b265cfb5091f991df07.png")

 **2.源码参考** 
https://github.com/PaddlePaddle/PaddleScience

 **3.工作目录** 
PaddlePaddle\paddlescience\PaddleScience-develop\examples\operator_learning

 **4.数据集获取** 

```
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepONet/antiderivative_unaligned_train.npz
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/DeepONet/antiderivative_unaligned_test.npz
```

 **5.训练代码** 

```
python deeponet.py

```

 **6.cpu训练脚本** 

```
python deeponet_cpu.py

```

 **7.不支持算子信息** 

 File "/root/miniconda3/envs/paddle_env/lib/python3.10/site-packages/paddle/base/dygraph/tensor_patch_methods.py", line 342, in backward
    core.eager.run_backward([self], grad_tensor, retain_graph)
![输入图片说明](https://foruda.gitee.com/images/1739958407685833232/ba9cae78_12173785.png "屏幕截图")

 **8.测试代码** 

```
python deeponet.py mode=eval EVAL.pretrained_model_path=https://paddle-org.bj.bcebos.com/paddlescience/models/deeponet/deeponet_pretrained.pdparams

```
 **9.结果评测** 
sdaa算子不支持，fall back到cpu可以正常运行，sdaa粗错如下：
![输入图片说明](https://foruda.gitee.com/images/1738901960070312843/9f9f562b_12173785.png "0196c01d846d0433a8fbc4b64b511df.png")