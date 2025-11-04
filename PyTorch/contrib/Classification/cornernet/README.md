# cornernet

## 1. 模型概述
CornerNet 是一种用于目标检测的深度学习模型，提出了一个创新的思路来解决物体的定位和检测问题。CornerNet 的核心思想是将目标的边界框表示为两个角点：左上角和右下角。通过检测这些角点，可以明确物体的位置信息，而不需要像传统方法那样使用多框回归（bounding box regression）来直接预测框的四个边界。它由 Shu Liu 等人提出，目的是在物体检测任务中提高精度和效率。

源码链接: https://github.com/open-mmlab/mmdetection

## 2. 快速开始

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 数据集准备

#### 2.2.1 数据集位置
/data/datasets/20241122/coco/


2. 安装python依赖
``` 
bash
# install requirements
pip install -r requirements.txt
```
### 2.4 启动训练
1. 在训练之前，我们需要对相关库进行必要的修改，这里修改的路径中的mjytorch为本人的环境名，需要更改。

首先安装需要的库：
``` 
pip install -r requirements.txt
```
接着请开始修改：
#### 1.修改了/root/miniconda3/envs/mjytorch/lib/python3.10/site-packages/mmengine/optim/optimizer/amp_optimizer_wrapper.py
第76-77行:
``` 
assert is_cuda_available() or is_npu_available() or is_mlu_available(
        ) or is_musa_available() or torch.sdaa.is_available(), (
```
#### 2.修改了/root/anaconda3/envs/mjytorch/lib/python3.10/site-packages/mmengine/device/utils.py
第53-58行：
```
mem = torch.sdaa.max_memory_allocated(device=device)
mem_mb = torch.tensor([int(mem) // (1024 * 1024)],
                          dtype=torch.int,
                          device=device)
torch.sdaa.reset_peak_memory_stats()
return int(mem_mb.item())
```
第63行:
 ```
 return torch.sdaa.is_available()
 ```
#### 3.修改了/root/miniconda3/envs/mjytorch/lib/python3.10/site-packages/mmengine/dist/utils.py
第130行：
```
torch.sdaa.set_device(local_rank)
```
第133行：
```
torch_dist.init_process_group(backend='tccl', **kwargs)
```
#### 4.修改了/root/anaconda3/envs/mjytorch/lib/python3.10/site-packages/mmengine/runner/runner.py
将877行的model = model.to(get_device()) 改为model = model.to("sdaa")
第2042行的device = get_device() 改为 device = "sdaa"

2. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Classification/cornernet
    ```

- 单机单核组
    ```
    torchrun tools/train.py configs/cornernet/cornernet_hourglass104_8xb6-210e-mstest_coco.py
    ```
- 单机单卡
    ```
    torchrun tools/train.py configs/cornernet/cornernet_hourglass104_8xb6-210e-mstest_coco.py --launcher pytorch --nproc_per_node 4
    ```


### 2.5 训练结果

| 芯片 |卡  | 模型 |  混合精度 |Batch size|Shape| 
|:-:|:-:|:-:|:-:|:-:|:-:|
|SDAA|1| cornernet |是|3|300*300|



