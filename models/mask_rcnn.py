from pathlib import Path
from tinygrad import nn

from tinygrad.tensor import Tensor
from extra.utils import download_file, get_child
from examples.serious_mnist import ConvBlock

class MaskRCNN:
    def __init__(self,  backbone, roi_heads, transform):
        self.backbone = backbone
        self.roi_head = roi_heads
        self.transform = transform

    def __call__(self, x):
        features = self.backbone(x)
        proposals = self.rpn(x, features)
        detections = self.roi_heads(features, proposals, images.image_sizes)
        detections = self.transform.postprocess(
            detections, images.image_sizes, original_image_sizes)  # type: ignore[operator

    def load_from_pretrained(self):
        return


class Backbone:
    def ___init__(self, model):
        self.model.fc = nn.Idenity()

    def __call__(self, x):
        out = self.model(x)
        return out

class RPNHead():
    def __init__(self,  in_channels, num_anchors, conv_depth=1):
        convs = []
        for _ in range(conv_depth):
            convs.append(ConvBlock(in_channels, in_channels, kernel_size=3, norm_layer=None))
        self.conv = nn.Sequential(*convs)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)
    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            t = self.conv(feature)
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg
if __name__ == "__main__":
    from models.resnet import ResNet18
    backbone = ResNet18()
    rpn = RPNHead()
    maskrcnn = MaskRCNN(backbone,rpn,None)
    x = Tensor.randn(1, 3, 244, 244)
    maskrcnn(x)
    