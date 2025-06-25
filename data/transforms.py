import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import cv2
import numpy as np

class RetinalTransforms:
    """
    Custom transforms for retinal images.

    Applies resizing, normalization, and data augmentation (random flip, rotation) for retinal images and masks.

    Args:
        image_size (tuple): Desired output image size (height, width).
        is_training (bool): Whether to apply training augmentations.

    Example:
        >>> transforms = RetinalTransforms((512, 512), is_training=True)
        >>> image, mask = transforms(image, mask)
    """
    
    def __init__(self, image_size=(512, 512), is_training=True):
        """
        Initialize the RetinalTransforms object.

        Args:
            image_size (tuple): Desired output image size (height, width).
            is_training (bool): Whether to apply training augmentations.
        """
        self.image_size = image_size
        self.is_training = is_training
        
    def __call__(self, image, mask=None):
        """
        Apply the defined transforms to an image (and optional mask).

        Args:
            image (Tensor): Input image tensor (C, H, W) or PIL Image.
            mask (Tensor or PIL Image, optional): Segmentation mask.

        Returns:
            tuple: (image, mask) after applying transforms.

        Example:
            >>> image, mask = transforms(image, mask)
        """
        # Resize
        image = F.resize(image, self.image_size)
        if mask is not None:
            mask = F.resize(mask, self.image_size, interpolation=transforms.InterpolationMode.NEAREST)
        
        if self.is_training:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                image = F.hflip(image)
                if mask is not None:
                    mask = F.hflip(mask)
            
            # Random rotation
            angle = torch.randint(-15, 15, (1,)).item()
            image = F.rotate(image, angle)
            if mask is not None:
                mask = F.rotate(mask, angle)
        
        # Normalize
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return image, mask