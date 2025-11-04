cd ..
export SDAA_VISIBLE_DEVICES=0,1,2,3
cfg_file=configs/bisenetv1_city.py
timeout 120m python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 6666 tools/train_amp.py --config $cfg_file 2>&1 | tee ./script/train_sdaa_3rd.log 