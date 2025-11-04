cd ../ 
pip install -r requirements.txt
pip install -v -e .
export SDAA_VISIBLE_DEVICES=0,1,2,3
export PADDLE_XCCL_BACKEND=sdaa
timeout 120m python -m  paddle.distributed.launch --devices=0,1,2,3 tools/train.py  \
       --config configs/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k.yml \
       --save_interval 50 \
       --save_dir output 
       
