cd ../ 
pip install -r requirements.txt
pip install -v -e .
export SDAA_VISIBLE_DEVICES=0,1,2,3
export PADDLE_XCCL_BACKEND=sdaa
timeout 120m python -m  paddle.distributed.launch --devices=0 tools/train.py  \
       --config contrib/PP-HumanSeg/configs/human_pp_humansegv1_lite.yml \
       --save_interval 50 \
       --save_dir output \
       --precision fp16
       
