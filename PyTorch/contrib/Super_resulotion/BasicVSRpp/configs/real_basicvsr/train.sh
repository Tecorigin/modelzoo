export SDAA_VISIBLE_DEVICES=0,1,2,3

# # single-gpu train
# python tools/train.py configs/real_basicvsr/realbasicvsr_c64b20-1x30x8_8xb1-lr5e-5-150k_reds.py

# multi-gpu train
./tools/dist_train.sh configs/real_basicvsr/realbasicvsr_c64b20-1x30x8_8xb1-lr5e-5-150k_reds.py 4