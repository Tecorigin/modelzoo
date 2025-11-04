# 代码来源
https://github.com/huggingface/diffusers.git
# 环境配置
# 拷贝太初已有的torch环境，创建新环境controlnet
    conda create --name new_env_name --clone existing_env_name
    例如：conda create --name controlnet --clone torch_env
    激活环境
    conda activate controlnet
#   1.首先下载代码以及配置模型
        pip install git+https://github.com/huggingface/diffusers.git transformers xformers==0.0.16 wandb

#   2.在hugging face上注册key值
        huggingface-cli login
        输入key值
#   3.在https://wandb.ai/注册key值
        wandb login
        输入key值
#   4.安装teco特定的accelerate库
    cd 进入保存teco-accelerate-dev_sdaa.tar文件的目录
    pip install teco-accelerate-dev_sdaa.tar
#   5.accelerate 配置
    accelerate config
    配置参数如下：compute_environment: LOCAL_MACHINE
                debug: true
                distributed_type: MULTI_SDAA    #选择多卡sdaa
                downcast_bf16: 'no'
                enable_cpu_affinity: false
                gpu_ids: 0,1,2,3       # 写卡号
                machine_rank: 0
                main_training_function: main
                mixed_precision: fp16       #混合精度
                num_machines: 1
                num_processes: 2
                rdzv_backend: static
                same_network: true
                tpu_env: []
                tpu_use_cluster: false
                tpu_use_sudo: false
                use_cpu: false
#    6.数据集准备
     fill50k/train/
                   -train.jsonl
                   -images/
                           -|***.jpg
                           -|***.jpg
                           ....

                   -conditioning_images/
                                        -|***.jpg
                                        -|***.jpg
                                        ......

#    7.train.jsonl 格式：
        {"text": "这里是输入的prompt", "image": "images/0.png", "conditioning_image": "conditioning_images/0.png"}
        {"text": "这里是输入的prompt2", "image": "images/1.png", "conditioning_image": "conditioning_images/1.png"}
        ......
        ......

#                 

# 数据集结构解释
# 运行指令
accelerate launch --multi_gpu train_controlnet.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
  --output_dir="model_out" \
  --train_data_dir="/data/datasets/20241122/fusing-fill50k/fill50k_data/train" \
  --conditioning_image_column="conditioning_image" \
  --image_column="image" \
  --caption_column="text" \
  --resolution=512 \
  --learning_rate=1e-5 \
  --validation_image "/data/datasets/20241122/fusing-fill50k/conditioning_images/0.png" \
  "/data/datasets/20241122/fusing-fill50k/conditioning_images/1.png" \
  "/data/datasets/20241122/fusing-fill50k/conditioning_images/2.png" \
  --validation_prompt "pale golden rod circle with old lace background" \
  "light coral circle with white background" \
  "aqua circle with light pink background" \
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