# 创建空测试脚本 test_empty.py
import torch
from mmdet.apis import init_detector

config = 'configs/retinanet/retinanet_r50_fpn_1x_coco.py'
checkpoint = 'work_dirs/latest.pth'
model = init_detector(config, checkpoint)

# 测试空输入
empty_img = torch.zeros((800, 1333, 3), dtype=torch.uint8)
result = model(empty_img)  # 应该返回空InstanceData不报错
print("测试通过!" if len(result.pred_instances) == 0 else "存在错误")