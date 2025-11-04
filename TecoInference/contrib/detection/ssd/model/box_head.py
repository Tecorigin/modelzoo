# Adapted to tecorigin hardware

from torch import nn
import torch.nn.functional as F
import torch

from .prior_box import PriorBox
from .box_predictor import SSDLiteBoxPredictor
from .box_utils import convert_locations_to_boxes, center_form_to_corner_form
from .container import Container
from .nms import batched_nms


class SSDBoxHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.predictor = SSDLiteBoxPredictor(cfg)
        self.priors = None

    def forward(self, features):
        cls_logits, bbox_pred = self.predictor(features)
        return self._forward_test(cls_logits, bbox_pred)

    def _forward_test(self, cls_logits, bbox_pred):
        if self.priors is None:
            self.priors = PriorBox(self.cfg)().to(bbox_pred.device)
        scores = F.softmax(cls_logits, dim=2)
        boxes = convert_locations_to_boxes(
            bbox_pred, self.priors, self.cfg.MODEL.CENTER_VARIANCE, self.cfg.MODEL.SIZE_VARIANCE
        )
        boxes = center_form_to_corner_form(boxes)
        detections = torch.cat((scores, boxes), dim=-1)
        return detections

