# eval_subset.py (已修正)
import json
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

# 1. 配置路径（请根据实际目录修改）
IMG_DIR    = Path("/data/teco-data/COCO/images/val2017")        # 验证集图片目录
ANN_FILE   = "/data/teco-data/COCO/annotations/instances_val2017.json"
MODEL_FILE = "yolov12n.onnx"

# 2. 载入 COCO GT，取前 N 张
coco = COCO(ANN_FILE)
img_ids = coco.getImgIds()[:100]  # 只取前 200 张
img_files = [IMG_DIR / coco.imgs[i]["file_name"] for i in img_ids]

# 3. 用 Ultralytics 推理 ONNX 模型（batch 与导出时匹配，可用 1）
model = YOLO(MODEL_FILE)
results = model.predict(
    source=img_files,
    imgsz=640,
    batch=1,
    conf=0.1,
    half=True,
    verbose=False
)

# 4. 收集检测结果，转换成 COCO 评估格式
detections = []
for image_id, r in zip(img_ids, results):
    # r.boxes.data: Tensor[N,6] = [x1,y1,x2,y2,conf,cls]
    data = r.boxes.data.cpu().numpy()  # shape (N,6)
    for x1, y1, x2, y2, conf, cls in data:
        detections.append({
            "image_id":    image_id,
            "category_id": int(cls),
            "bbox":        [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score":       float(conf)
        })

# 5. 写入 JSON 并评估
ahead_file = "preds.json"
with open(ahead_file, "w") as f:
    json.dump(detections, f)

coco_dt   = coco.loadRes(ahead_file)
coco_eval = COCOeval(coco, coco_dt, iouType="bbox")
coco_eval.params.imgIds = img_ids
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()  # 打印 mAP50 / mAP50-95 / precision / recall
