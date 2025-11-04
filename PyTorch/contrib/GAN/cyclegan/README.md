#  CycleGAN
## 1. 模型概述
CycleGAN 是一种无监督的图像到图像转换生成对抗网络。它的架构设计旨在解决不同图像域之间的转换问题，比如将夏季风景图转换为冬季风景图，或是将照片风格转换为绘画风格等，且无需成对的训练数据。​
该模型主要由两个生成器和两个判别器组成。两个生成器分别负责将一个域的图像转换到另一个域，例如生成器 G 将图像从 A 域转换到 B 域，生成器 F 则执行反向操作，将 B 域图像转换回 A 域。判别器的作用是判断生成的图像是否来自真实数据分布。CycleGAN 引入循环一致性损失（Cycle Consistency Loss）这一核心概念，通过强制转换后的图像再转换回原图像时尽可能与原始图像相似，以此确保生成的图像在语义上的合理性。例如，将马的图像转换为斑马图像后，再从斑马图像转换回马的图像，应与最初的马图像相近。
## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装
请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集
#### 2.2.1 获取数据集
常用的数据集地址如下：
链接: https://pan.baidu.com/s/1xng_uQjyG-8CFMktEXRdEg 提取码: grtm
下载horse2zebra的数据集
#### 2.2.2 处理数据集
将下载的horse2zebra数据中的A、B类分别放到cyclegan/datasets的A、B文件夹下
运行根目录下面的txt_annotation.py，生成train_lines.txt，保证train_lines.txt内部是有文件路径内容的。
处理后数据集结构：
```angular2html
    |-- A
        |-- .......jpg
    |-- B
        |-- .......jpg       

    |-- train_lines.txt
``` 


### 2.3 构建环境
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```
2. 修改虚拟环境的代码（由于torch.pad操作sdaa不支持）
在/root/miniconda3/envs/cyclegan/lib/python3.10/site-packages/torch/nn/modules/padding.py，
将_ReflectionPadNd：
```angular2html
    class _ReflectionPadNd(Module):
        __constants__ = ['padding']
        padding: Sequence[int]

        def forward(self, input: Tensor) -> Tensor:
            return F.pad(input, self.padding, 'reflect')
```

改为：
```angular2html
    import torch
    def aligned_cpu_tensor(shape, dtype=torch.float32, alignment=64):
        numel = torch.Size(shape).numel()
        element_size = torch.tensor([], dtype=dtype).element_size()
        bytes_need = numel * element_size
        bytes_need = (bytes_need + alignment - 1) // alignment * alignment  # 对齐到alignment字节
        
        # 分配对齐内存并截断到精确长度
        storage = torch.ByteStorage.from_buffer(bytearray(bytes_need), 'cpu')
        tensor = torch.tensor(storage, dtype=torch.uint8)
        return tensor[:numel * element_size].view(dtype).view(shape)

    class _ReflectionPadNd(Module):
        __constants__ = ['padding']
        padding: Sequence[int]
        def forward(self, x):
            input_cpu = x.cpu()
            h_pad = self.padding[2] + self.padding[3]
            w_pad = self.padding[0] + self.padding[1]
            output_shape = (
                input_cpu.size(0),
                input_cpu.size(1),
                input_cpu.size(2) + h_pad,
                input_cpu.size(3) + w_pad
            )
            output_cpu = aligned_cpu_tensor(output_shape, dtype=input_cpu.dtype)
            F.pad(input_cpu, self.padding, 'reflect')
            return output_cpu.contiguous().to(x.device)
        def extra_repr(self) -> str:
            return f'{self.padding}'
```
3. 所需安装的包
    ```
    pip install -r requirements.txt
    ```
    
### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/Pytorch/contrib/GAN/cyclegan
    ```
   
2. 运行训练。

   ```
   torchrun --nproc_per_node 4 train.py
   ```


### 2.5 训练结果
### 2.5 训练结果
模型训练3h，得到结果如下  
|加速卡数量|模型|Epoch|Batch size|D_Loss_A|D_Loss_A|
| :-: | :-: | :-: | :-: | :-: | :-: |  
|4|CycleGAN|27|4|0.01|0.05|
