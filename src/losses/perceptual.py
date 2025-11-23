import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models.feature_extraction import create_feature_extractor

class VGGPerceptual(nn.Module):
    """
    Perceptual loss on RGB using VGG16 features.
    Uses torchvision's create_feature_extractor for robust taps.
    Inputs must be float tensors in [0,1] on the same device.
    """
    def __init__(self, layers=("relu1_2","relu2_2","relu3_3"), weights=(1.0, 0.5, 0.2)):
        super().__init__()
        self.layers = tuple(layers)
        self.weights = tuple(weights)

        backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features.eval()
        node_map = {"relu1_2":"3", "relu2_2":"8", "relu3_3":"15", "relu4_3":"22", "relu5_3":"29"}
        return_nodes = { node_map[name]: name for name in self.layers }
        self.extractor = create_feature_extractor(backbone, return_nodes=return_nodes)
        for p in self.extractor.parameters(): p.requires_grad = False

    @staticmethod
    def _norm_imagenet(x: torch.Tensor) -> torch.Tensor:
        mean = x.new_tensor([0.485, 0.456, 0.406])[None,:,None,None]
        std  = x.new_tensor([0.229, 0.224, 0.225])[None,:,None,None]
        return (x - mean) / std

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0,1); y = y.clamp(0,1)
        x = self._norm_imagenet(x); y = self._norm_imagenet(y)
        fx = self.extractor(x)
        fy = self.extractor(y)
        loss = 0.0
        for name, w in zip(self.layers, self.weights):
            loss = loss + w*F.l1_loss(fx[name], fy[name])
        return loss