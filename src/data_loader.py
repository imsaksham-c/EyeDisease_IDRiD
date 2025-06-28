import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
from src import config
import albumentations as A
from albumentations.pytorch import ToTensorV2

class IDRiDDataset(Dataset):
    """
    Custom PyTorch Dataset for the preprocessed IDRiD data.
    Loads images, segmentation masks, and classification labels.
    """
    def __init__(self, df, data_path, transforms=None):
        self.df = df
        self.data_path = data_path
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # --- Load Image ---
        image_path = os.path.join(self.data_path, row['image_path'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- Load Mask ---
        mask_path_rel = row['mask_path']
        if pd.isna(mask_path_rel):
            # If no mask exists, create a blank one.
            mask = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, config.NUM_CLASSES_SEGMENTATION), dtype=np.float32)
        else:
            mask_path = os.path.join(self.data_path, mask_path_rel)
            mask = np.load(mask_path).astype(np.float32)
            
        # --- Load Label ---
        classification_label = torch.tensor(row['dr_grade'], dtype=torch.long)
        
        # --- Apply Transformations ---
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            
            # --- FIX: Manually permute the mask dimensions ---
            # albumentations' ToTensorV2 keeps the mask as (H, W, C).
            # PyTorch's loss functions expect (C, H, W).
            # This line reorders the dimensions to be compatible.
            if isinstance(mask, torch.Tensor):
                 mask = mask.permute(2, 0, 1)

        return {
            "image": image,
            "segmentation_mask": mask,
            "classification_label": classification_label
        }

def get_loaders():
    """Returns training and validation data loaders."""
    
    train_df = pd.read_csv(config.TRAIN_CSV)
    test_df = pd.read_csv(config.TEST_CSV)

    # Define augmentation pipelines
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transforms = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    train_dataset = IDRiDDataset(
        df=train_df,
        data_path=config.DATA_PATH,
        transforms=train_transforms,
    )
    
    test_dataset = IDRiDDataset(
        df=test_df,
        data_path=config.DATA_PATH,
        transforms=val_transforms,
    )
    
    # Use half of the available CPU cores for data loading to avoid bottlenecks
    # Temporarily disable multiprocessing to avoid PyTorch issues
    num_workers = 0  # int(os.cpu_count() / 2) if os.cpu_count() else 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader
