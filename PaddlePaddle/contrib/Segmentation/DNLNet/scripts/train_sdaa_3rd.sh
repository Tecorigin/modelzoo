cd ../ 
pip install -r requirements.txt
pip install -v -e .
export SDAA_VISIBLE_DEVICES=0,1,2,3
export PADDLE_XCCL_BACKEND=sdaa
export CUSTOM_DEVICE_BLACK_LIST=conv2d
timeout 120m python -m  paddle.distributed.launch --devices=0,1,2,3 tools/train.py  \
       --config ./configs/dnlnet/dnlnet_resnet50_os8_cityscapes_1024x512_80k.yml \
       --save_interval 50 \
       --save_dir output 
       # --precision fp16 | tee ./scripts/train_sdaa_3rd.log 
       
