# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

from argument import parse_args
import os
from pathlib import Path

if __name__ == '__main__':
    args = parse_args()
    
    model_name = args.model_name
    epochs = args.epochs
    bs = args.batch_size
    lr = args.lr
    nnode = args.nnode
    node_rank = args.node_rank
    nproc_per_node = args.nproc_per_node
    data_dir = args.data_dir
    
    project_path = str(Path(__file__).resolve().parents[1])
        
    if model_name == 'albu_example':
        config_path = f'{project_path}/configs/albu_example/mask-rcnn_r50_fpn_albu-1x_coco.py'
    elif model_name == 'atss':
        config_path = f'{project_path}/configs/atss/atss_r50_fpn_1x_coco.py'
    elif model_name == 'autoassign':
        config_path = f'{project_path}/configs/autoassign/autoassign_r50-caffe_fpn_1x_coco.py'
    elif model_name == 'boxinst':
        config_path = f'{project_path}/configs/boxinst/boxinst_r50_fpn_ms-90k_coco.py'
    elif model_name == 'carafe':
        config_path = f'{project_path}/configs/carafe/mask-rcnn_r50_fpn-carafe_1x_coco.py'
    elif model_name == 'cascade_rcnn':
        config_path = f'{project_path}/configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py'
    elif model_name == 'cascade_rpn':
        config_path = f'{project_path}/configs/cascade_rpn/cascade-rpn_r50-caffe_fpn_1x_coco.py'
    elif model_name == 'centernet':
        config_path = f'{project_path}/configs/centernet/centernet-update_r50-caffe_fpn_ms-1x_coco.py'
    elif model_name == 'centripetalnet':
        config_path = f'{project_path}/configs/centripetalnet/centripetalnet_hourglass104_16xb6-crop511-210e-mstest_coco.py'
    elif model_name == 'condinst':
        config_path = f'{project_path}/configs/condinst/condinst_r50_fpn_ms-poly-90k_coco_instance.py'
    elif model_name == 'conditional_detr':
        config_path = f'{project_path}/configs/conditional_detr/conditional-detr_r50_8xb2-50e_coco.py'
    elif model_name == 'convnext':
        config_path = f'{project_path}/configs/convnext/mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco.py'
    elif model_name == 'cornernet':
        config_path = f'{project_path}/configs/cornernet/cornernet_hourglass104_8xb6-210e-mstest_coco.py'
    elif model_name == 'dab_detr':
        config_path = f'{project_path}/configs/dab_detr/dab-detr_r50_8xb2-50e_coco.py'
    elif model_name == 'dcn':
        config_path = f'{project_path}/configs/dcn/faster-rcnn_r50_fpn_dpool_1x_coco.py'
    elif model_name == 'dcnv2':
        config_path = f'{project_path}/configs/dcnv2/faster-rcnn_r50_fpn_mdpool_1x_coco.py'
    elif model_name == 'ddod':
        config_path = f'{project_path}/configs/ddod/ddod_r50_fpn_1x_coco.py'
    elif model_name == 'ddq':
        config_path = f'{project_path}/configs/ddq/ddq-detr-4scale_r50_8xb2-12e_coco.py'
    elif model_name == 'deformable_detr':
        config_path = f'{project_path}/configs/deformable_detr/deformable-detr_r50_16xb2-50e_coco.py'
    elif model_name == 'detectors':
        config_path = f'{project_path}/configs/detectors/htc_r50-sac_1x_coco.py'
    elif model_name == 'detr':
        config_path = f'{project_path}/configs/detr/detr_r50_8xb2-150e_coco.py'
    elif model_name == 'dino':
        config_path = f'{project_path}/configs/dino/dino-4scale_r50_8xb2-12e_coco.py'
    elif model_name == 'double_heads':
        config_path = f'{project_path}/configs/double_heads/dh-faster-rcnn_r50_fpn_1x_coco.py'
    # elif model_name == 'dsdl':
    #     config_path = f'{project_path}/configs/dsdl/coco.py'
    elif model_name == 'dyhead':
        config_path = f'{project_path}/configs/dyhead/atss_r50_fpn_dyhead_1x_coco.py'
    elif model_name == 'dynamic_rcnn':
        config_path = f'{project_path}/configs/dynamic_rcnn/dynamic-rcnn_r50_fpn_1x_coco.py'
    elif model_name == 'efficientnet':
        config_path = f'{project_path}/configs/efficientnet/retinanet_effb3_fpn_8xb4-crop896-1x_coco.py'
    elif model_name == 'empirical_attention':
        config_path = f'{project_path}/configs/empirical_attention/faster-rcnn_r50-attn0010_fpn_1x_coco.py'
    elif model_name == 'fast_rcnn':
        config_path = f'{project_path}/configs/fast_rcnn/fast-rcnn_r50_fpn_1x_coco.py'
    elif model_name == 'faster_rcnn':
        config_path = f'{project_path}/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
    # elif model_name == 'fcos':
    #     config_path = f'{project_path}/configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py'
    elif model_name == 'foveabox':
        config_path = f'{project_path}/configs/foveabox/fovea_r50_fpn_4xb4-1x_coco.py'
    elif model_name == 'fpg':
        config_path = f'{project_path}/configs/fpg/retinanet_r50_fpg_crop640_50e_coco.py'
    elif model_name == 'free_anchor':
        config_path = f'{project_path}/configs/free_anchor/freeanchor_r50_fpn_1x_coco.py'
    elif model_name == 'fsaf':
        config_path = f'{project_path}/configs/fsaf/fsaf_r50_fpn_1x_coco.py'
    elif model_name == 'gcnet':
        config_path = f'{project_path}/configs/gcnet/mask-rcnn_r50-syncbn_fpn_1x_coco.py'
    elif model_name == 'gfl':
        config_path = f'{project_path}/configs/gfl/gfl_r50_fpn_1x_coco.py'
    elif model_name == 'ghm':
        config_path = f'{project_path}/configs/ghm/retinanet_r50_fpn_ghm-1x_coco.py'
    elif model_name == 'glip':
        config_path = f'{project_path}/configs/glip/glip_atss_swin-l_fpn_dyhead_16xb2_ms-2x_funtune_coco.py'
    elif model_name == 'gn':
        config_path = f'{project_path}/configs/gn/mask-rcnn_r50_fpn_gn-all_2x_coco.py'
    elif model_name == 'gn+ws':
        config_path = f'{project_path}/configs/gn+ws/faster-rcnn_r50_fpn_gn-ws-all_1x_coco.py'
    elif model_name == 'grid_rcnn':
        config_path = f'{project_path}/configs/grid_rcnn/grid-rcnn_r50_fpn_gn-head_1x_coco.py'
    elif model_name == 'groie':
        config_path = f'{project_path}/configs/groie/mask-rcnn_r50_fpn_groie_1x_coco.py'
    elif model_name == 'grounding_dino':
        config_path = f'{project_path}/configs/grounding_dino/grounding_dino_r50_scratch_8xb2_1x_coco.py'
    elif model_name == 'guided_anchoring':
        config_path = f'{project_path}/configs/guided_anchoring/ga-rpn_r50_fpn_1x_coco.py'
    elif model_name == 'hrnet':
        config_path = f'{project_path}/configs/hrnet/htc_hrnetv2p-w40_20e_coco.py'
    elif model_name == 'htc':
        config_path = f'{project_path}/configs/htc/htc_r50_fpn_1x_coco.py'
    elif model_name == 'instaboost':
        config_path = f'{project_path}/configs/instaboost/mask-rcnn_r50_fpn_instaboost-4x_coco.py'
    elif model_name == 'lad':
        config_path = f'{project_path}/configs/lad/lad_r50-paa-r101_fpn_2xb8_coco_1x.py'
    elif model_name == 'ld':
        config_path = f'{project_path}/configs/ld/ld_r50-gflv1-r101_fpn_1x_coco.py'
    elif model_name == 'legacy_1.x':
        config_path = f'{project_path}/configs/legacy_1.x/mask-rcnn_r50_fpn_1x_coco_v1.py'
    elif model_name == 'libra_rcnn':
        config_path = f'{project_path}/configs/libra_rcnn/libra-fast-rcnn_r50_fpn_1x_coco.py'
    elif model_name == 'mask2former':
        config_path = f'{project_path}/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco.py'
    elif model_name == 'mask_rcnn':
        config_path = f'{project_path}/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'
    elif model_name == 'maskformer':
        config_path = f'{project_path}/configs/maskformer/maskformer_r50_ms-16xb1-75e_coco.py'
    elif model_name == 'misc':
        config_path = f'{project_path}/configs/misc/d2_mask-rcnn_r50-caffe_fpn_ms-90k_coco.py'
    elif model_name == 'ms_rcnn':
        config_path = f'{project_path}/configs/ms_rcnn/ms-rcnn_r50_fpn_1x_coco.py'
    elif model_name == 'nas_fcos':
        config_path = f'{project_path}/configs/nas_fcos/nas-fcos_r50-caffe_fpn_nashead-gn-head_4xb4-1x_coco.py'
    elif model_name == 'nas_fpn':
        config_path = f'{project_path}/configs/nas_fpn/retinanet_r50_fpn_crop640-50e_coco.py'
    elif model_name == 'paa':
        config_path = f'{project_path}/configs/paa/paa_r50_fpn_1x_coco.py'
    elif model_name == 'pafpn':
        config_path = f'{project_path}/configs/pafpn/faster-rcnn_r50_pafpn_1x_coco.py'
    elif model_name == 'panoptic_fpn':
        config_path = f'{project_path}/configs/panoptic_fpn/panoptic-fpn_r50_fpn_1x_coco.py'
    elif model_name == 'pascal_voc':
        config_path = f'{project_path}/configs/pascal_voc/faster-rcnn_r50_fpn_1x_voc0712-cocofmt.py'
    elif model_name == 'pisa':
        config_path = f'{project_path}/configs/pisa/retinanet-r50_fpn_pisa_1x_coco.py'
    elif model_name == 'point_rend':
        config_path = f'{project_path}/configs/point_rend/point-rend_r50-caffe_fpn_ms-1x_coco.py'
    elif model_name == 'pvt':
        config_path = f'{project_path}/configs/pvt/retinanet_pvt-m_fpn_1x_coco.py'
    elif model_name == 'queryinst':
        config_path = f'{project_path}/configs/queryinst/queryinst_r50_fpn_1x_coco.py'
    elif model_name == 'regnet':
        config_path = f'{project_path}/configs/regnet/mask-rcnn_regnetx-4GF_fpn_1x_coco.py'
    elif model_name == 'reppoints':
        config_path = f'{project_path}/configs/reppoints/reppoints-moment_r50_fpn_1x_coco.py'
    elif model_name == 'res2net':
        config_path = f'{project_path}/configs/res2net/htc_res2net-101_fpn_20e_coco.py'
    elif model_name == 'resnest':
        config_path = f'{project_path}/configs/resnest/mask-rcnn_s50_fpn_syncbn-backbone+head_ms-1x_coco.py'
    elif model_name == 'resnet_strikes_back':
        config_path = f'{project_path}/configs/resnet_strikes_back/mask-rcnn_r50-rsb-pre_fpn_1x_coco.py'
    elif model_name == 'retinanet':
        config_path = f'{project_path}/configs/retinanet/retinanet_r50_fpn_1x_coco.py'
    elif model_name == 'rpn':
        config_path = f'{project_path}/configs/rpn/rpn_r50_fpn_1x_coco.py'
    elif model_name == 'rtmdet':
        config_path = f'{project_path}/configs/rtmdet/rtmdet_l_8xb32-300e_coco.py'
    elif model_name == 'sabl':
        config_path = f'{project_path}/configs/sabl/sabl-retinanet_r50_fpn_1x_coco.py'
    elif model_name == 'scnet':
        config_path = f'{project_path}/configs/scnet/scnet_r50_fpn_1x_coco.py'
    elif model_name == 'scratch':
        config_path = f'{project_path}/configs/scratch/mask-rcnn_r50-scratch_fpn_gn-all_6x_coco.py'
    elif model_name == 'selfsup_pretrain':
        config_path = f'{project_path}/configs/selfsup_pretrain/mask-rcnn_r50-swav-pre_fpn_1x_coco.py'
    elif model_name == 'simple_copy_paste':
        config_path = f'{project_path}/configs/simple_copy_paste/mask-rcnn_r50_fpn_rpn-2conv_4conv1fc_syncbn-all_32xb2-ssj-90k_coco.py'
    elif model_name == 'soft_teacher':
        config_path = f'{project_path}/configs/soft_teacher/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.1-coco.py'
    elif model_name == 'solo':
        config_path = f'{project_path}/configs/solo/solo_r50_fpn_1x_coco.py'
    elif model_name == 'solov2':
        config_path = f'{project_path}/configs/solov2/solov2_r50_fpn_1x_coco.py'
    elif model_name == 'sparse_rcnn':
        config_path = f'{project_path}/configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py'
    elif model_name == 'ssd':
        config_path = f'{project_path}/configs/ssd/ssd512_coco.py'
    elif model_name == 'strong_baselines':
        config_path = f'{project_path}/configs/strong_baselines/mask-rcnn_r50_fpn_rpn-2conv_4conv1fc_syncbn-all_lsj-50e_coco.py'
    elif model_name == 'swin':
        config_path = f'{project_path}/configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
    elif model_name == 'timm_example':
        config_path = f'{project_path}/configs/timm_example/retinanet_timm-tv-resnet50_fpn_1x_coco.py'
    elif model_name == 'tood':
        config_path = f'{project_path}/configs/tood/tood_r50_fpn_1x_coco.py'
    elif model_name == 'tridentnet':
        config_path = f'{project_path}/configs/tridentnet/tridentnet_r50-caffe_1x_coco.py'
    # elif model_name == 'vfnet':
    #     config_path = f'{project_path}/configs/vfnet/vfnet_r50_fpn_1x_coco.py'
    elif model_name == 'yolact':
        config_path = f'{project_path}/configs/yolact/yolact_r50_1xb8-55e_coco.py'
    elif model_name == 'yolo':
        config_path = f'{project_path}/configs/yolo/yolov3_d53_8xb8-320-273e_coco.py'
    elif model_name == 'yolof':
        config_path = f'{project_path}/configs/yolof/yolof_r50-c5_8xb8-1x_coco.py'
    elif model_name == 'yolox':
        config_path = f'{project_path}/configs/yolox/yolox_s_8xb8-300e_coco.py'
    else:
        raise ValueError('Model name not recognized.')
    
    cmd = f' bash {project_path}/tools/dist_train.sh {config_path} {nproc_per_node} --batch-size {bs} --lr {lr} --epochs {epochs} --data-dir {data_dir}'
    print('本次运行命令',cmd)
    os.system(cmd)
    