import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationExpert(nn.Module):
    """
    Expert module for disease classification.

    Applies global average pooling and fully connected layers to produce class logits for disease grading.

    Args:
        feature_dim (int): Input feature dimension.
        num_classes (int): Number of disease classes.
        hidden_dim (int): Hidden layer dimension.
        dropout (float): Dropout rate.

    Example:
        >>> expert = ClassificationExpert(2048, 5)
        >>> logits = expert(torch.randn(8, 2048, 16, 16))
    """
    
    def __init__(self, feature_dim, num_classes, hidden_dim=256, dropout=0.2):
        """
        Initialize the classification expert.

        Args:
            feature_dim (int): Input feature dimension.
            num_classes (int): Number of disease classes.
            hidden_dim (int): Hidden layer dimension.
            dropout (float): Dropout rate.
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, features):
        """
        Forward pass for classification.

        Args:
            features (Tensor): Input feature map (B, C, H, W).

        Returns:
            Tensor: Class logits (B, num_classes).

        Example:
            >>> logits = expert(features)
        """
        return self.classifier(features)

class SegmentationExpert(nn.Module):
    """
    Expert module for lesion segmentation.

    U-Net style decoder for producing binary segmentation masks from feature maps.

    Args:
        feature_dim (int): Input feature dimension.
        dropout (float): Dropout rate.

    Example:
        >>> expert = SegmentationExpert(2048)
        >>> mask = expert(torch.randn(8, 2048, 16, 16))
    """
    
    def __init__(self, feature_dim, dropout=0.2):
        """
        Initialize the segmentation expert.

        Args:
            feature_dim (int): Input feature dimension.
            dropout (float): Dropout rate.
        """
        super().__init__()
        
        # Decoder for segmentation
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        """
        Forward pass for segmentation.

        Args:
            features (Tensor): Input feature map (B, C, H, W).

        Returns:
            Tensor: Segmentation mask (B, 1, H_out, W_out), values in [0, 1].

        Example:
            >>> mask = expert(features)
        """
        return self.decoder(features)

class GeneralExpert(nn.Module):
    """
    General purpose expert module.

    Applies convolutional processing for feature refinement or auxiliary tasks.

    Args:
        feature_dim (int): Input feature dimension.
        hidden_dim (int): Hidden layer dimension.
        dropout (float): Dropout rate.

    Example:
        >>> expert = GeneralExpert(2048)
        >>> out = expert(torch.randn(8, 2048, 16, 16))
    """
    
    def __init__(self, feature_dim, hidden_dim=256, dropout=0.2):
        """
        Initialize the general expert.

        Args:
            feature_dim (int): Input feature dimension.
            hidden_dim (int): Hidden layer dimension.
            dropout (float): Dropout rate.
        """
        super().__init__()
        
        self.processor = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, features):
        """
        Forward pass for general expert.

        Args:
            features (Tensor): Input feature map (B, C, H, W).

        Returns:
            Tensor: Processed feature map.

        Example:
            >>> out = expert(features)
        """
        return self.processor(features)