export SDAA_VISIBLE_DEVICES=0,1
export SDAA_LAUNCH_BLOCKING=1
export TORCH_SDAA_LOG_LEVEL=debug 
# bash ./tools/dist_train.sh configs/basicvsr/basicvsr_2xb4_reds4.py 2
python tools/train.py configs/basicvsr/basicvsr_2xb4_reds4.py | tee sdaa.log
# export SDAA_VISIBLE_DEVICES=-1
# python tools/train.py configs/basicvsr/basicvsr_2xb4_reds4.py