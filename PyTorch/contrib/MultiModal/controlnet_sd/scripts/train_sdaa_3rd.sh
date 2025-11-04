#!/bin/bash

# 定义日志文件路径
cd ..
LOG_FILE="scripts/train_sdaa_3rd.log"
export HF_ENDPOINT="https://hf-mirror.com"
# 使用 accelerate launch 启动训练脚本并将输出重定向到日志文件
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
  --mixed_precision=no > "${LOG_FILE}" 2>&1