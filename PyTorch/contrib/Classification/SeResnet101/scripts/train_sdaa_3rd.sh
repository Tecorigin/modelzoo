cd ..
export TORCH_HOME=/data/datasets/checkpoints/
SDAA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh configs/seresnet/seresnet101_8xb32_in1k.py 4 | tee ./scripts/train_sdaa_3rd.log 
