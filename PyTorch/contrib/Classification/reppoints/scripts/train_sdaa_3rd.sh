cd ..
# pip install -r requirements.txt
torchrun tools/train.py configs/reppoints/reppoints-bbox_r50_fpn-gn_head-gn-grid_1x_coco.py --launcher pytorch --amp