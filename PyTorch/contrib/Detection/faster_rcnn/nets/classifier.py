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
import warnings

import torch
from torch import nn
from torchvision.ops import RoIPool

warnings.filterwarnings("ignore")

class VGG16RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(VGG16RoIHead, self).__init__()
        self.classifier = classifier
        #--------------------------------------#
        #   对ROIPooling后的的结果进行回归预测
        #--------------------------------------#
        self.cls_loc    = nn.Linear(4096, n_class * 4)
        #-----------------------------------#
        #   对ROIPooling后的的结果进行分类
        #-----------------------------------#
        self.score      = nn.Linear(4096, n_class)
        #-----------------------------------#
        #   权值初始化
        #-----------------------------------#
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)
        
    def forward(self, x, rois, roi_indices, img_size):
        n, _, _, _ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()
        rois        = torch.flatten(rois, 0, 1)
        roi_indices = torch.flatten(roi_indices, 0, 1)

        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0,2]] = rois[:, [0,2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1,3]] = rois[:, [1,3]] / img_size[0] * x.size()[2]

        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)
        #-----------------------------------#
        #   利用建议框对公用特征层进行截取
        #-----------------------------------#
        pool = self.roi(x, indices_and_rois)
        #-----------------------------------#
        #   利用classifier网络进行特征提取
        #-----------------------------------#
        pool = pool.view(pool.size(0), -1)
        #--------------------------------------------------------------#
        #   当输入为一张图片的时候，这里获得的f7的shape为[300, 4096]
        #--------------------------------------------------------------#
        fc7 = self.classifier(pool)

        roi_cls_locs    = self.cls_loc(fc7)
        roi_scores      = self.score(fc7)

        roi_cls_locs    = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores      = roi_scores.view(n, -1, roi_scores.size(1))
        return roi_cls_locs, roi_scores

class Resnet50RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(Resnet50RoIHead, self).__init__()
        self.classifier = classifier
        #--------------------------------------#
        #   对ROIPooling后的的结果进行回归预测
        #--------------------------------------#
        self.cls_loc = nn.Linear(2048, n_class * 4)
        #-----------------------------------#
        #   对ROIPooling后的的结果进行分类
        #-----------------------------------#
        self.score = nn.Linear(2048, n_class)
        #-----------------------------------#
        #   权值初始化
        #-----------------------------------#
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)

    def forward(self, x, rois, roi_indices, img_size):
        n, _, _, _ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()
        rois        = torch.flatten(rois, 0, 1)
        roi_indices = torch.flatten(roi_indices, 0, 1)
        
        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0,2]] = rois[:, [0,2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1,3]] = rois[:, [1,3]] / img_size[0] * x.size()[2]

        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)
        #-----------------------------------#
        #   利用建议框对公用特征层进行截取
        #-----------------------------------#
        pool = self.roi(x, indices_and_rois)
        #-----------------------------------#
        #   利用classifier网络进行特征提取
        #-----------------------------------#
        fc7 = self.classifier(pool)
        #--------------------------------------------------------------#
        #   当输入为一张图片的时候，这里获得的f7的shape为[300, 2048]
        #--------------------------------------------------------------#
        fc7 = fc7.view(fc7.size(0), -1)

        roi_cls_locs    = self.cls_loc(fc7)
        roi_scores      = self.score(fc7)
        roi_cls_locs    = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores      = roi_scores.view(n, -1, roi_scores.size(1))
        return roi_cls_locs, roi_scores

def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
