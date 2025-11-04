cd ..
export SDAA_VISIBLE_DEVICES=0,1,2,3
export TORCH_HOME=/data/datasets/checkpoints/
timeout 120m sh tools/dist_train.sh  ./configs/ocrnet/ocrnet_hr18_4xb2-40k_cityscapes-512x1024.py 4 | tee ./scripts/train_sdaa_3rd.log