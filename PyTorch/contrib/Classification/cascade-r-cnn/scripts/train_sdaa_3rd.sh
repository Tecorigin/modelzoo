cd ..
# pip install -r requirements.txt
torchrun tools/train.py configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py --launcher pytorch --nproc_per_node 4