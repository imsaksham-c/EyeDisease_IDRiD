import torch
import torch.nn as nn
import torchvision.models as models


class SharedBackbone(nn.Module):
    """
    Shared feature extraction backbone.

    Uses a ResNet-based architecture (ResNet-50 or ResNet-34) to extract features from input images.

    Args:
        backbone_name (str): Name of the ResNet backbone ('resnet50' or 'resnet34').
        pretrained (bool): Whether to use ImageNet pretrained weights.

    Attributes:
        features (nn.Sequential): Feature extractor up to the last convolutional block.
        feature_dim (int): Output feature dimension.

    Example:
        >>> backbone = SharedBackbone('resnet50', pretrained=True)
        >>> features = backbone(torch.randn(1, 3, 512, 512))
    """

    def __init__(self, backbone_name="resnet50", pretrained=True):
        """
        Initialize the shared backbone.

        Args:
            backbone_name (str): Name of the ResNet backbone ('resnet50' or 'resnet34').
            pretrained (bool): Whether to use ImageNet pretrained weights.
        Raises:
            ValueError: If an unsupported backbone is specified.
        """
        super().__init__()

        if backbone_name == "resnet50":
            backbone = models.resnet50(pretrained=pretrained)
            self.features = nn.Sequential(*list(backbone.children())[:-2])
            self.feature_dim = 2048
        elif backbone_name == "resnet34":
            backbone = models.resnet34(pretrained=pretrained)
            self.features = nn.Sequential(*list(backbone.children())[:-2])
            self.feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward(self, x):
        """
        Forward pass through the backbone.

        Args:
            x (Tensor): Input image tensor of shape (B, 3, H, W).

        Returns:
            Tensor: Extracted feature map.

        Example:
            >>> features = backbone(torch.randn(1, 3, 512, 512))
        """
        return self.features(x)
