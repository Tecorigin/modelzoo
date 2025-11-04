export SDAA_VISIBLE_DEVICES=0,1,2,3
# export SDAA_LAUNCH_BLOCKING=1
# export TORCH_SDAA_RUNTIME_AUTOFALLBACK=1
# export TORCH_SDAA_FALLBACK_OPS=fused_adam
# export TORCH_SDAA_LOG_LEVEL=debug 
# python tools/train.py configs/basicvsr_pp/basicvsr-pp_c64n7_8xb1-600k_reds4.py | tee train_sdaa_3rd.log
./tools/dist_train.sh configs/basicvsr_pp/basicvsr-pp_c64n7_8xb1-600k_reds4.py 4 | tee train_sdaa_3rd.log