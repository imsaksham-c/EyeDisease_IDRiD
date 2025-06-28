"""
data_utils.py
-------------
Provides dataset classes and utility functions for loading, preprocessing, and augmenting the IDRiD dataset for both segmentation and classification tasks.
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import torchvision.transforms as transforms
from .dataset_reorganiser import DatasetReorganizer

class EyeDiseaseDataset(Dataset):
    """
    PyTorch Dataset for Eye Disease Multi-task Learning.
    Handles both classification and segmentation tasks.
    """
    def __init__(self, data_list, transform=None, task_type='both'):
        self.data_list = data_list
        self.transform = transform
        self.task_type = task_type
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # Load image
        image = Image.open(item['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        sample = {
            'image': image,
            'image_id': item['image_id'],
            'retinopathy_grade': item['retinopathy_grade'],
            'macular_edema_risk': item['macular_edema_risk'],
            'has_segmentation': item['has_segmentation']
        }
        
        # Load segmentation masks if available and needed
        if item['has_segmentation'] and item['segmentation_masks'] and self.task_type in ['both', 'segmentation']:
            masks = self._load_segmentation_masks(item['segmentation_masks'], image.shape[-2:])
            sample['segmentation_mask'] = masks
        else:
            sample['segmentation_mask'] = None
        
        return sample
    
    def _load_segmentation_masks(self, mask_paths, target_size):
        """
        Loads and combines all available segmentation masks for an image.
        Args:
            mask_paths (dict): Dict of mask_type -> mask_path
            target_size (tuple): (height, width) to resize masks
        Returns:
            torch.Tensor: Combined mask tensor
        """
        combined_mask = np.zeros((target_size[0], target_size[1]), dtype=np.float32)
        for mask_type, mask_path in mask_paths.items():
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = cv2.resize(mask, (target_size[1], target_size[0]))
                    mask = (mask > 127).astype(np.float32)
                    combined_mask = np.maximum(combined_mask, mask)
        return torch.from_numpy(combined_mask).unsqueeze(0)

def create_data_transforms():
    """
    Returns torchvision transforms for training and validation.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform 