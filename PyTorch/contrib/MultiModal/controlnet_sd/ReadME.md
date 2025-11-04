#  Controlnet
## 1. 模型概述
ControlNet 是一种基于神经网络的插件，专为 StableDiffusion 设计的辅助式神经网络模型结构。它采用在 StableDiffusion 模型基础上添加辅助模块的架构，通过复制出 “锁定” 副本与 “可训练” 副本，在 “可训练” 副本施加控制条件，并将其结果与原始 StableDiffusion 模型结果相加输出。这种结构设计，为模型带来了强大的条件控制能力。​
ControlNet 引入了独特的控制机制，能够接收涂鸦、边缘映射、分割映射、pose 关键点等多种条件输入，并将这些条件融入到扩散模型中。在文本到图像生成过程里，该机制发挥作用，使得生成图像能高度契合用户预期，极大地提升了生成图像的精准度与可控性。与传统图像到图像生成方法相比，在稳定性上实现了质的飞跃，允许用户在生成过程中对各类参数进行微调。​


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
fusing-fill50k数据集下载地址如下：
https://huggingface.co/datasets/fusing/fill50k

#### 2.2.2 处理数据集
解压数据集，将其按格式放入文件datasets中，格式目录如工作目录所示

#### LJSpeech数据集和权重目录结构：
```angular2html
|-- fill50k_data
    |-- images
        |--...png
    |-- conditioning_images               
        |--...png
    |-- train.jsonl
    |-- fill50k.py

```

### 2.3 构建环境
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```
   
2. 所需安装的包
    ```
    pip install -r requirements.txt
    ```
    注：accelerate库是taichu自定义版本的，解压后直接pip下载
### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/MultiModal/controlnet_sd
    ```
   
2. 运行训练。
  
- 多卡
   ```
   export HF_ENDPOINT="https://hf-mirror.com"
   accelerate launch --multi_gpu diffusers-main/examples/controlnet/train_controlnet.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
    --output_dir="model_out" \
    --train_data_dir="/data/datasets/fusing-fill50k/fill50k_data/train" \
    --conditioning_image_column="conditioning_image" \
    --image_column="image" \
    --caption_column="text" \
    --resolution=512 \
    --learning_rate=1e-5 \
    --validation_image "/data/datasets/fusing-fill50k/conditioning_images/0.png" \
    --validation_prompt "pale golden rod circle with old lace background" \
    --train_batch_size=1 \
    --num_train_epochs=3 \
    --tracker_project_name="controlnet" \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps=5000 \
    --validation_steps=5000 \
    --report_to wandb \
    --push_to_hub \
    --gradient_accumulation_steps=4 \
    --mixed_precision=no 

   ```

### 2.5 训练结果
模型训练2h，得到结果如下  
|加速卡数量|模型|Epoch|Batch size|Loss|  
| :-: | :-: | :-: | :-: | :-: | 
|4|controlnet|1|1|0.0075|



