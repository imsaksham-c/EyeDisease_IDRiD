import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np

SEGMENTATION_TYPES = [
    ("Microaneurysms", "MA"),
    ("Haemorrhages", "HE"),
    ("Hard Exudates", "EX"),
    ("Soft Exudates", "SE"),
    ("Optic Disc", "OD"),
]

class IDRiDDataset(Dataset):
    """
    IDRiD Dataset for multi-task learning (segmentation, grading, or both).

    Args:
        data_root (str): Root directory of the dataset (should be 'dataset/').
        split (str): 'train' or 'test'.
        transforms (callable, optional): Transformations to apply to images and masks.
        task (str): 'segmentation', 'grading', or 'multi' (default: 'multi').
    """
    def __init__(self, data_root, split="train", transforms=None, task="multi"):
        self.data_root = data_root
        self.split = split
        self.transforms = transforms
        self.task = task
        self.samples = []

        # Map split to folder names
        if split.lower() in ["train", "training"]:
            seg_img_dir = os.path.join(data_root, "A. Segmentation/1. Original Images/a. Training Set")
            seg_mask_dir = os.path.join(data_root, "A. Segmentation/2. All Segmentation Groundtruths/a. Training Set")
            grad_img_dir = os.path.join(data_root, "B. Disease Grading/1. Original Images/a. Training Set")
            grad_label_csv = os.path.join(data_root, "B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv")
        else:
            seg_img_dir = os.path.join(data_root, "A. Segmentation/1. Original Images/b. Testing Set")
            seg_mask_dir = os.path.join(data_root, "A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set")
            grad_img_dir = os.path.join(data_root, "B. Disease Grading/1. Original Images/b. Testing Set")
            grad_label_csv = os.path.join(data_root, "B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv")

        # Load grading labels
        grading_labels = {}
        if os.path.exists(grad_label_csv):
            df = pd.read_csv(grad_label_csv)
            for _, row in df.iterrows():
                # Remove spaces in column names if present
                img_name = str(row[0]).strip()
                grade = int(row[1]) if not pd.isna(row[1]) else None
                grading_labels[img_name] = grade

        # Collect all unique image names from both tasks
        seg_images = set(f[:-4] for f in os.listdir(seg_img_dir) if f.endswith('.jpg')) if os.path.exists(seg_img_dir) else set()
        grad_images = set(f[:-4] for f in os.listdir(grad_img_dir) if f.endswith('.jpg')) if os.path.exists(grad_img_dir) else set()
        all_images = seg_images | grad_images | set(grading_labels.keys())

        for img_name in sorted(all_images):
            sample = {"image_name": img_name}
            # Segmentation image path
            seg_img_path = os.path.join(seg_img_dir, f"{img_name}.jpg")
            if os.path.exists(seg_img_path):
                sample["seg_img_path"] = seg_img_path
            # Grading image path
            grad_img_path = os.path.join(grad_img_dir, f"{img_name}.jpg")
            if os.path.exists(grad_img_path):
                sample["grad_img_path"] = grad_img_path
            # Segmentation masks (dict of type: path)
            masks = {}
            for lesion, suffix in SEGMENTATION_TYPES:
                mask_path = os.path.join(seg_mask_dir, f"{SEGMENTATION_TYPES.index((lesion, suffix))+1}. {lesion}", f"{img_name}_{suffix}.tif")
                if os.path.exists(mask_path):
                    masks[suffix] = mask_path
            if masks:
                sample["masks"] = masks
            # Grading label
            if img_name in grading_labels:
                sample["grade"] = grading_labels[img_name]
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        result = {"image_name": sample["image_name"]}

        # Load image (prefer segmentation image, else grading image)
        img_path = sample.get("seg_img_path") or sample.get("grad_img_path")
        image = Image.open(img_path).convert("RGB")
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # Load masks (if available)
        masks = None
        if "masks" in sample:
            masks = {}
            for suffix, mask_path in sample["masks"].items():
                mask = Image.open(mask_path).convert("L")
                mask = torch.from_numpy(np.array(mask)).float() / 255.0
                mask = (mask > 0.5).float()
                masks[suffix] = mask
        # If no masks, create dummy masks
        if masks is None:
            masks = {suffix: torch.zeros(image.shape[1], image.shape[2]) for _, suffix in SEGMENTATION_TYPES}

        # Load grading label (if available)
        grade = sample.get("grade", None)

        # Apply transforms
        if self.transforms:
            image, masks = self.transforms(image, masks)

        # Return according to task
        if self.task == "segmentation":
            result["image"] = image
            result["masks"] = masks
        elif self.task == "grading":
            result["image"] = image
            result["class_label"] = torch.tensor(grade if grade is not None else -1, dtype=torch.long)
        else:  # multi-task
            result["image"] = image
            result["masks"] = masks
            result["class_label"] = torch.tensor(grade if grade is not None else -1, dtype=torch.long)
        return result
