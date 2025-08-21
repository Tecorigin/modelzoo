# mmpretrain/datasets/coco_caption.py
# Copyright (c) OpenMMLab. All rights reserved.
from typing import List
import mmengine
from mmengine.dataset import BaseDataset
from mmengine.fileio import get_file_backend
from mmpretrain.registry import DATASETS

@DATASETS.register_module()
class COCOCaption(BaseDataset):
    """COCO Caption dataset supporting standard COCO caption JSON format."""

    def load_data_list(self) -> List[dict]:
        # 确保 data_prefix 中有 img_path
        if 'img_path' not in self.data_prefix:
            raise KeyError(
                "COCOCaption dataset requires `data_prefix=dict(img_path=...)` "
                "in config, but not found."
            )

        img_prefix = self.data_prefix['img_path']
        file_backend = get_file_backend(img_prefix)

        coco_json = mmengine.load(self.ann_file)
        id2filename = {img['id']: img['file_name'] for img in coco_json['images']}

        data_list = []
        for ann in coco_json['annotations']:
            img_file = id2filename[ann['image_id']]
            data_info = {
                'image_id': str(ann['image_id']),
                'img_path': file_backend.join_path(img_prefix, img_file),
                'gt_caption': ann['caption'],
            }
            data_list.append(data_info)

        return data_list
