cd ../ 
export SDAA_VISIBLE_DEVICES=0,1,2,3
export PADDLE_XCCL_BACKEND=sdaa
timeout 120m python -m  paddle.distributed.launch --devices=0,1 tools/train.py  \
       --config configs/unet_plusplus/unet_plusplus_cityscapes_1024x512_160k.yml \
       --save_interval 50 \
       --save_dir output \
       --precision fp16 
       
