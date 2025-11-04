import re
import json
import os

import torch
import torch_sdaa
import torch.distributed as dist

import utils
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json',
            'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json'}
    filenames = {'val':'captions_val2017.json','test':'captions_val2017.json'}    
    
    #download_url(urls[split],coco_gt_root)
    annotation_file = os.path.join(coco_gt_root,filenames[split])
    
    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
    
    return coco_eval

coco_gt_root = "/data/datasets/20241122/coco/annotations"
val_result_file = "/root/cas/tww/BLIP/output/Caption_coco/result/val_epoch0.json"
coco_val = coco_caption_eval(coco_gt_root,val_result_file,'val')
print("======",coco_val.eval.items())
'''for k, v in coco_val.eval.items():
    print(k,v)'''