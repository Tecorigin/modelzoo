import json
import os
from ultralytics.data.converter import convert_coco

json_path = '/data/datasets/coco/annotations'
output_path = '/data/datasets/coco/convert'
# json_path = '/root/data/datasets/test/annotations'
# output_path = '/root/data/datasets/test/convert'
print("start convert")
convert_coco(json_path, output_path)