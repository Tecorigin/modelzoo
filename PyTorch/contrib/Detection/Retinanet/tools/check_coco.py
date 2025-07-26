#!/usr/bin/env python3
import json
import numpy as np
from collections import defaultdict

def quick_coco_check(ann_file):
    """快速检查COCO标注文件的常见问题"""
    with open(ann_file) as f:
        data = json.load(f)
    
    # 基础统计
    print(f"\n{'='*40}\n数据集概览\n{'='*40}")
    print(f"图片数量: {len(data['images'])}")
    print(f"标注数量: {len(data['annotations'])}")
    print(f"类别数量: {len(data['categories'])}")

    # 构建图片-标注映射
    ann_dict = defaultdict(list)
    for ann in data['annotations']:
        ann_dict[ann['image_id']].append(ann)
    
    # 1. 检查空标注图片
    empty_imgs = [img for img in data['images'] if not ann_dict.get(img['id'])]
    print(f"\n{'='*40}\n空标注图片\n{'='*40}")
    print(f"数量: {len(empty_imgs)}")
    if empty_imgs:
        print("示例空图片:", empty_imgs[0]['file_name'])

    # 2. 检查无效标注
    invalid_anns = {
        'width<=0': [],
        'height<=0': [],
        'area<=0': [],
        '越界坐标': []
    }
    
    for ann in data['annotations']:
        x, y, w, h = ann['bbox']
        img = next((i for i in data['images'] if i['id'] == ann['image_id']), None)
        
        if w <= 0: invalid_anns['width<=0'].append(ann)
        if h <= 0: invalid_anns['height<=0'].append(ann)
        if ann['area'] <= 0: invalid_anns['area<=0'].append(ann)
        if img and (x + w > img['width'] or y + h > img['height']):
            invalid_anns['越界坐标'].append(ann)
    
    print(f"\n{'='*40}\n无效标注统计\n{'='*40}")
    for k, v in invalid_anns.items():
        print(f"{k}: {len(v)}处")

    # 3. 类别分布检查
    cat_dist = defaultdict(int)
    for ann in data['annotations']:
        cat_dist[ann['category_id']] += 1
    
    print(f"\n{'='*40}\n类别分布(TOP5)\n{'='*40}")
    for cat_id, count in sorted(cat_dist.items(), key=lambda x: -x[1])[:5]:
        cat_name = next((c['name'] for c in data['categories'] if c['id'] == cat_id), '未知')
        print(f"{cat_name}({cat_id}): {count}")

    # 4. 标注面积分布
    areas = [ann['area'] for ann in data['annotations']]
    print(f"\n{'='*40}\n标注面积分布\n{'='*40}")
    print(f"最小面积: {np.min(areas):.2f}")
    print(f"中位数面积: {np.median(areas):.2f}")
    print(f"最大面积: {np.max(areas):.2f}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("用法: python check_coco.py <标注文件路径>")
        sys.exit(1)
    quick_coco_check(sys.argv[1])