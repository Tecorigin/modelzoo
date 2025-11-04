cd ../
pip install -r requirements.txt
# 修改data/coco.yaml中的coco数据集路径
python -m torch.distributed.launch --nproc_per_node 4 tools/train.py --batch 128 --img-size 416 --conf configs/yolov6_lite/yolov6_lite_s.py --data data/coco.yaml --epoch 400 --device 0,1,2,3 --name yolov6_lite_s_coco > yolov6_lite_s.log 2>&1 &
# 画loss对比图
cd scripts
python loss_check.py