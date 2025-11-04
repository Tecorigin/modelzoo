cd ..
# pip install -r requirements.txt
torchrun tools/train.py configs/rtmdet/rtmdet_s_8xb32-300e_coco.py --launcher pytorch --amp