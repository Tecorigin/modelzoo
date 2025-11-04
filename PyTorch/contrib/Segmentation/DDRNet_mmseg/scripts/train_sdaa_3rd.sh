cd ..
export SDAA_VISIBLE_DEVICES=0,1,2,3
export TORCH_HOME=/data/datasets/checkpoints/
timeout 120m sh tools/dist_train.sh configs/ddrnet/ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024.py 4 | tee ./scripts/train_sdaa_3rd.log