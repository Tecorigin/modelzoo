export SDAA_VISIBLE_DEVICES=0,1,2,3
# export SDAA_LAUNCH_BLOCKING=1
# export TORCH_SDAA_RUNTIME_AUTOFALLBACK=1
# export TORCH_SDAA_FALLBACK_OPS=fused_adam
# export TORCH_SDAA_LOG_LEVEL=debug 
bash ./tools/dist_train.sh configs/basicvsr/basicvsr_2xb4_reds4.py 4 | tee train_sdaa_3rd.log
# python tools/train.py configs/basicvsr/basicvsr_2xb4_reds4.py | tee sdaa.log
# export SDAA_VISIBLE_DEVICES=-1
# python -X faulthandler tools/train.py configs/basicvsr/basicvsr_2xb4_reds4.py \
# python tools/train.py configs/basicvsr/basicvsr_2xb4_reds4.py \
# | tee basicvsr_error.log