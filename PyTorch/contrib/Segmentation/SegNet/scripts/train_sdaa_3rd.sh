cd ../CamVid
export SDAA_VISIBLE_DEVICES=0,1,2,3
export TORCH_HOME=/data/datasets/checkpoints/
timeout 120m torchrun --nproc_per_node 4 Train_SegNet.py 2>&1 | tee ../scripts/train_sdaa_3rd.log 