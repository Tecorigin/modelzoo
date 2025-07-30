# Adapted to tecorigin hardware

from torch import nn

from .mobilenetv3 import MobileNetV3
from .box_head import SSDBoxHead
from .defaults import _C as cfg

class SSDDetector(nn.Module):
    def __init__(self,):
        super().__init__()
        cfg.merge_from_file('model/mobilenet_v3_ssd320_voc0712.yaml')
        self.backbone = MobileNetV3()
        self.box_head = SSDBoxHead(cfg)

    def forward(self, images):
        features = self.backbone(images)
        detections = self.box_head(features)
        return detections