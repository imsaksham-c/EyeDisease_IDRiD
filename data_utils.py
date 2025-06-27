import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import torchvision.transforms as transforms

class DatasetReorganizer:
    """
    Utility class to reorganize the dataset for multi-task learning.
    Prepares combined data entries for both segmentation and grading tasks.
    """
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.seg_train_path = os.path.join(dataset_path, "A. Segmentation/1. Original Images/a. Training Set")
        self.seg_test_path = os.path.join(dataset_path, "A. Segmentation/1. Original Images/b. Testing Set")
        self.seg_gt_train_path = os.path.join(dataset_path, "A. Segmentation/2. All Segmentation Groundtruths/a. Training Set")
        self.seg_gt_test_path = os.path.join(dataset_path, "A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set")
        self.grade_train_path = os.path.join(dataset_path, "B. Disease Grading/1. Original Images/a. Training Set")
        self.grade_test_path = os.path.join(dataset_path, "B. Disease Grading/1. Original Images/b. Testing Set")
        self.grade_labels_path = os.path.join(dataset_path, "B. Disease Grading/2. Groundtruths")
    
    def reorganize_data(self):
        """
        Combines segmentation and grading data into unified train/test lists.
        Returns:
            dict: {'train': [...], 'test': [...], 'segmentation_only': ...}
        """
        seg_data = self._prepare_segmentation_data()
        grade_data = self._prepare_grading_data()
        
        combined_train_data = []
        combined_test_data = []
        
        # Create mapping from segmentation data
        seg_train_map = {item['image_id']: item for item in seg_data['train']}
        seg_test_map = {item['image_id']: item for item in seg_data['test']}
        
        # For grading data, create combined entries
        for item in grade_data['train']:
            # Normalize image_id to match segmentation naming (strip leading zeros)
            image_id = item['image_id'].lstrip('0').replace('IDRiD_', '')
            image_id = f"IDRiD_{int(image_id)}"  # e.g., IDRiD_10
            seg_item = seg_train_map.get(image_id, None)
            
            combined_item = {
                'image_path': item['image_path'],
                'image_id': image_id,
                'retinopathy_grade': item['retinopathy_grade'],
                'macular_edema_risk': item['macular_edema_risk'],
                'has_segmentation': seg_item is not None,
                'segmentation_masks': seg_item['masks'] if seg_item else None
            }
            combined_train_data.append(combined_item)
        
        for item in grade_data['test']:
            # Normalize image_id to match segmentation naming (strip leading zeros)
            image_id = item['image_id'].lstrip('0').replace('IDRiD_', '')
            image_id = f"IDRiD_{int(image_id)}"
            seg_item = seg_test_map.get(image_id, None)
            
            combined_item = {
                'image_path': item['image_path'],
                'image_id': image_id,
                'retinopathy_grade': item['retinopathy_grade'],
                'macular_edema_risk': item['macular_edema_risk'],
                'has_segmentation': seg_item is not None,
                'segmentation_masks': seg_item['masks'] if seg_item else None
            }
            combined_test_data.append(combined_item)
        
        return {
            'train': combined_train_data,
            'test': combined_test_data,
            'segmentation_only': seg_data
        }
    
    def _prepare_segmentation_data(self):
        """
        Prepares segmentation data for train and test sets.
        Returns:
            dict: {'train': [...], 'test': [...]}
        """
        train_data = []
        test_data = []
        
        # Process training data
        if os.path.exists(self.seg_train_path):
            for img_file in os.listdir(self.seg_train_path):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    image_id = os.path.splitext(img_file)[0]
                    img_path = os.path.join(self.seg_train_path, img_file)
                    masks = self._get_segmentation_masks(image_id, self.seg_gt_train_path)
                    
                    train_data.append({
                        'image_path': img_path,
                        'image_id': image_id,
                        'masks': masks
                    })
        
        # Process test data
        if os.path.exists(self.seg_test_path):
            for img_file in os.listdir(self.seg_test_path):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    image_id = os.path.splitext(img_file)[0]
                    img_path = os.path.join(self.seg_test_path, img_file)
                    masks = self._get_segmentation_masks(image_id, self.seg_gt_test_path)
                    
                    test_data.append({
                        'image_path': img_path,
                        'image_id': image_id,
                        'masks': masks
                    })
        
        return {'train': train_data, 'test': test_data}
    
    def _get_segmentation_masks(self, image_id, gt_path):
        """
        Returns a dict of available mask paths for a given image_id.
        """
        masks = {}
        mask_types = ['1. Microaneurysms', '2. Haemorrhages', '3. Hard Exudates', 
                      '4. Soft Exudates', '5. Optic Disc']
        mask_suffixes = ['_MA.tif', '_HE.tif', '_EX.tif', '_SE.tif', '_OD.tif']
        
        for mask_type, suffix in zip(mask_types, mask_suffixes):
            mask_path = os.path.join(gt_path, mask_type, f"{image_id}{suffix}")
            if os.path.exists(mask_path):
                masks[mask_type] = mask_path
        
        return masks
    
    def _prepare_grading_data(self):
        """
        Prepares grading (classification) data for train and test sets.
        Returns:
            dict: {'train': [...], 'test': [...]}
        """
        train_data = []
        test_data = []
        
        # Load training labels
        train_labels_path = os.path.join(self.grade_labels_path, "a. IDRiD_Disease Grading_Training Labels.csv")
        if os.path.exists(train_labels_path):
            train_df = pd.read_csv(train_labels_path)
            for _, row in train_df.iterrows():
                image_name = row['Image name']
                img_path = os.path.join(self.grade_train_path, f"{image_name}.jpg")
                if os.path.exists(img_path):
                    train_data.append({
                        'image_path': img_path,
                        'image_id': image_name,
                        'retinopathy_grade': int(row['Retinopathy grade']),
                        'macular_edema_risk': int(row['Risk of macular edema'])
                    })
        
        # Load test labels
        test_labels_path = os.path.join(self.grade_labels_path, "b. IDRiD_Disease Grading_Testing Labels.csv")
        if os.path.exists(test_labels_path):
            test_df = pd.read_csv(test_labels_path)
            for _, row in test_df.iterrows():
                image_name = row['Image name']
                img_path = os.path.join(self.grade_test_path, f"{image_name}.jpg")
                if os.path.exists(img_path):
                    test_data.append({
                        'image_path': img_path,
                        'image_id': image_name,
                        'retinopathy_grade': int(row['Retinopathy grade']),
                        'macular_edema_risk': int(row['Risk of macular edema'])
                    })
        
        return {'train': train_data, 'test': test_data}

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