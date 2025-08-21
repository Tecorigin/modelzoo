# # import json

# # json_path = '/data/teco-data/coco/annotations/coco_karpathy_train.json'
# # output_path = '/data/teco-data/coco/annotations/coco_karpathy_train_2017.json'

# # with open(json_path, 'r', encoding='utf-8') as f:
# #     data = json.load(f)

# # for item in data:
# #     if 'image' in item:
# #         # 替换 image 字段中的 train2014 为 train2017
# #         item['image'] = item['image'].replace('train2014', 'train2017')
# #     if 'image_id' in item:
# #         # 可选，替换image_id里可能有的年份
# #         item['image_id'] = item['image_id'].replace('train2014', 'train2017')

# # with open(output_path, 'w', encoding='utf-8') as f:
# #     json.dump(data, f, ensure_ascii=False, indent=2)

# # print(f"保存完成，路径：{output_path}")

# import json
# import os

# json_path = '/data/teco-data/coco/annotations/coco_karpathy_val_2017.json'
# output_path = '/data/teco-data/coco/annotations/coco_karpathy_val_new.json'

# with open(json_path, 'r', encoding='utf-8') as f:
#     data = json.load(f)

# for item in data:
#     if 'image' in item:
#         # 只保留文件名部分
#         filename = os.path.basename(item['image'])
#         # 如果文件名前缀是 COCO_ 开头，去掉前缀部分，只保留数字和后缀
#         if filename.startswith('COCO_'):
#             # 例如 'COCO_val2014_000000522418.jpg'
#             # 取最后一个下划线后的部分
#             filename = filename.split('_')[-1]
#         item['image'] = filename
#     if 'image_id' in item:
#         # 可选，不修改 image_id 也可以
#         pass

# with open(output_path, 'w', encoding='utf-8') as f:
#     json.dump(data, f, ensure_ascii=False, indent=2)

# print(f"保存完成，路径：{output_path}")

import json

with open('/data/teco-data/coco/annotations/coco_karpathy_train_new.json', 'r') as f:
    data = json.load(f)

for item in data:
    if '191761' in item['image']:
        print(item)
        break
