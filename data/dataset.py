import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np


class IDRiDDataset(Dataset):
    """
    IDRiD Dataset for multi-task learning.

    This dataset handles loading images, segmentation masks, and classification labels for the IDRiD dataset.
    It supports multi-task learning for both disease grading (classification) and lesion segmentation.

    Args:
        data_root (str): Root directory of the dataset.
        split (str): Dataset split to use ('train', 'val', or 'test').
        transforms (callable, optional): Transformations to apply to images and masks.

    Example:
        >>> dataset = IDRiDDataset('data/', split='train')
        >>> sample = dataset[0]
        >>> image, mask, label = sample['image'], sample['mask'], sample['class_label']
    """

    def __init__(self, data_root, split="train", transforms=None):
        """
        Initialize the IDRiDDataset.

        Args:
            data_root (str): Root directory of the dataset.
            split (str): Dataset split to use ('train', 'val', or 'test').
            transforms (callable, optional): Transformations to apply to images and masks.
        """
        self.data_root = data_root
        self.split = split
        self.transforms = transforms

        # Load classification labels
        self.class_df = pd.read_csv(
            os.path.join(data_root, f"classification_labels/{split}.csv")
        )

        # Get image paths
        self.image_dir = os.path.join(data_root, f"images/{split}")
        self.mask_dir = os.path.join(data_root, f"segmentation_masks/{split}")

        self.samples = []
        for idx, row in self.class_df.iterrows():
            image_name = row["image_name"]
            image_path = os.path.join(self.image_dir, f"{image_name}.jpg")
            mask_path = os.path.join(self.mask_dir, f"{image_name}_mask.png")

            if os.path.exists(image_path):
                self.samples.append(
                    {
                        "image_path": image_path,
                        "mask_path": mask_path if os.path.exists(mask_path) else None,
                        "class_label": int(row["grade"]),
                        "image_name": image_name,
                    }
                )

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'image' (Tensor): The image tensor (C, H, W).
                - 'mask' (Tensor): The segmentation mask tensor (H, W).
                - 'class_label' (Tensor): The disease grade label.
                - 'has_mask' (Tensor): 1.0 if mask exists, else 0.0.
                - 'image_name' (str): The image file name (without extension).

        Example:
            >>> sample = dataset[0]
            >>> image = sample['image']
            >>> mask = sample['mask']
            >>> label = sample['class_label']
        """
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # Load mask
        mask = None
        if sample["mask_path"] and os.path.exists(sample["mask_path"]):
            mask = Image.open(sample["mask_path"]).convert("L")
            mask = torch.from_numpy(np.array(mask)).float() / 255.0
            mask = (mask > 0.5).float()  # Binarize
        else:
            # Create dummy mask if not available
            mask = torch.zeros(image.shape[1], image.shape[2])

        # Apply transforms
        if self.transforms:
            image, mask = self.transforms(image, mask)

        return {
            "image": image,
            "mask": mask,
            "class_label": torch.tensor(sample["class_label"], dtype=torch.long),
            "has_mask": torch.tensor(1.0 if sample["mask_path"] else 0.0),
            "image_name": sample["image_name"],
        }
