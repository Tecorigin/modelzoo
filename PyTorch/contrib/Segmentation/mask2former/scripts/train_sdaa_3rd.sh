mim install mmdet
cd ..
export SDAA_VISIBLE_DEVICES=0,1,2,3
timeout 120m sh tools/dist_train.sh ./configs/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024.py 4 | tee ./scripts/train_sdaa_3rd.log