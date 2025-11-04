cd ..
export SDAA_VISIBLE_DEVICES=2,3
export TORCH_HOME=/data/datasets/checkpoints/
timeout 120m sh tools/dist_train.sh configs/apcnet/apcnet_r50-d8_4xb2-40k_cityscapes-512x1024.py 2 | tee ./scripts/train_sdaa_3rd.log