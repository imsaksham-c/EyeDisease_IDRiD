# Dataset handling for IDRiD dataset
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

class IDRiDDataset(Dataset):
    """IDRiD Dataset for multi-task learning with balanced sampling support"""
    
    def __init__(self, config, split='train', transforms=None, balance_sampling=True):
        self.config = config
        self.split = split
        self.transforms = transforms
        self.image_size = config.image_size
        self.balance_sampling = balance_sampling
        
        # Initialize data lists
        self.images = []
        self.segmentation_masks = []
        self.classification_labels = []
        self.task_types = []  # 'seg', 'cls', or 'both'
        self.sample_weights = []  # For balanced sampling
        
        # Dataset statistics
        self.dataset_stats = {
            'total_samples': 0,
            'segmentation_samples': 0,
            'classification_samples': 0,
            'dr_grades': Counter(),
            'dme_risks': Counter(),
            'lesion_counts': defaultdict(int),
            'class_weights': {}
        }
        
        self._load_data()
        self._calculate_class_weights()
        
    def _load_data(self):
        """Load data from IDRiD dataset structure"""
        # Load segmentation data
        seg_img_path = os.path.join(self.config.seg_data_path, "1. Original Images")
        seg_mask_path = os.path.join(self.config.seg_data_path, "2. All Segmentation Groundtruths")
        
        # Load classification data
        cls_img_path = os.path.join(self.config.cls_data_path, "1. Original Images")
        cls_labels_path = os.path.join(self.config.cls_data_path, "2. Groundtruths")
        
        if self.split == 'train':
            seg_folder = "a. Training Set"
            cls_folder = "a. Training Set"
            cls_labels_file = "a. IDRiD_Disease Grading_Training Labels.csv"
        else:
            seg_folder = "b. Testing Set"
            cls_folder = "b. Testing Set"
            cls_labels_file = "b. IDRiD_Disease Grading_Testing Labels.csv"
        
        # Load segmentation samples
        seg_img_folder = os.path.join(seg_img_path, seg_folder)
        seg_mask_folder = os.path.join(seg_mask_path, seg_folder)
        
        if os.path.exists(seg_img_folder):
            for img_file in sorted(os.listdir(seg_img_folder)):
                if img_file.endswith('.jpg'):
                    img_path = os.path.join(seg_img_folder, img_file)
                    img_id = img_file.replace('.jpg', '')
                    
                    # Load corresponding masks
                    masks = self._load_segmentation_masks(seg_mask_folder, img_id)
                    
                    self.images.append(img_path)
                    self.segmentation_masks.append(masks)
                    self.classification_labels.append(None)
                    self.task_types.append('seg')
                    
                    # Count lesions for this sample
                    for i, mask in enumerate(masks[1:], 1):  # Skip background
                        if mask.sum() > 0:
                            self.dataset_stats['lesion_counts'][i] += 1
        
        # Load classification samples
        cls_img_folder = os.path.join(cls_img_path, cls_folder)
        cls_labels_filepath = os.path.join(cls_labels_path, cls_labels_file)
        
        if os.path.exists(cls_img_folder) and os.path.exists(cls_labels_filepath):
            cls_df = pd.read_csv(cls_labels_filepath)
            
            for _, row in cls_df.iterrows():
                img_name = f"{row['Image name']}.jpg"
                img_path = os.path.join(cls_img_folder, img_name)
                
                if os.path.exists(img_path):
                    dr_grade = int(row['Retinopathy grade'])
                    dme_risk = int(row['Risk of macular edema '])
                    
                    self.images.append(img_path)
                    self.segmentation_masks.append(None)
                    self.classification_labels.append((dr_grade, dme_risk))
                    self.task_types.append('cls')
                    
                    # Count class distributions
                    self.dataset_stats['dr_grades'][dr_grade] += 1
                    self.dataset_stats['dme_risks'][dme_risk] += 1
        
        # Update total counts
        self.dataset_stats['total_samples'] = len(self.images)
        self.dataset_stats['segmentation_samples'] = sum(1 for t in self.task_types if t == 'seg')
        self.dataset_stats['classification_samples'] = sum(1 for t in self.task_types if t == 'cls')
        
        logger.info(f"Loaded {len(self.images)} samples for {self.split}")
        logger.info(f"Segmentation samples: {self.dataset_stats['segmentation_samples']}")
        logger.info(f"Classification samples: {self.dataset_stats['classification_samples']}")
        
        # Log class distributions
        if self.dataset_stats['dr_grades']:
            logger.info(f"DR Grade distribution: {dict(self.dataset_stats['dr_grades'])}")
        if self.dataset_stats['dme_risks']:
            logger.info(f"DME Risk distribution: {dict(self.dataset_stats['dme_risks'])}")
        if self.dataset_stats['lesion_counts']:
            logger.info(f"Lesion distribution: {dict(self.dataset_stats['lesion_counts'])}")
    
    def _calculate_class_weights(self):
        """Calculate class weights for balanced sampling"""
        if not self.balance_sampling:
            self.sample_weights = [1.0] * len(self.images)
            return
        
        # Calculate weights based on class frequencies
        weights = []
        
        for i, task_type in enumerate(self.task_types):
            weight = 1.0
            
            if task_type == 'cls' and self.classification_labels[i] is not None:
                dr_grade, dme_risk = self.classification_labels[i]
                
                # DR grade weight (inverse frequency)
                if self.dataset_stats['dr_grades']:
                    dr_weight = len(self.images) / (len(self.dataset_stats['dr_grades']) * self.dataset_stats['dr_grades'][dr_grade])
                    weight *= dr_weight
                
                # DME risk weight (inverse frequency)
                if self.dataset_stats['dme_risks']:
                    dme_weight = len(self.images) / (len(self.dataset_stats['dme_risks']) * self.dataset_stats['dme_risks'][dme_risk])
                    weight *= dme_weight
            
            elif task_type == 'seg' and self.segmentation_masks[i] is not None:
                masks = self.segmentation_masks[i]
                
                # Calculate lesion presence weight
                lesion_presence = 0
                for j, mask in enumerate(masks[1:], 1):  # Skip background
                    if mask.sum() > 0:
                        lesion_presence += 1
                
                # Weight based on lesion complexity (more lesions = higher weight)
                if lesion_presence > 0:
                    seg_weight = 1.0 + (lesion_presence * 0.2)  # Boost samples with lesions
                    weight *= seg_weight
                
                # Balance between seg and cls tasks
                if self.dataset_stats['segmentation_samples'] > 0 and self.dataset_stats['classification_samples'] > 0:
                    task_balance = self.dataset_stats['classification_samples'] / self.dataset_stats['segmentation_samples']
                    weight *= task_balance
            
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        self.sample_weights = [w / total_weight for w in weights]
        
        # Store class weights for loss functions
        self.dataset_stats['class_weights'] = {
            'dr_weights': self._calculate_class_weights_from_counter(self.dataset_stats['dr_grades']),
            'dme_weights': self._calculate_class_weights_from_counter(self.dataset_stats['dme_risks']),
            'lesion_weights': self._calculate_lesion_weights()
        }
        
        logger.info(f"Sample weights calculated - Min: {min(weights):.3f}, Max: {max(weights):.3f}, Mean: {np.mean(weights):.3f}")
    
    def _calculate_class_weights_from_counter(self, counter):
        """Calculate class weights from a counter"""
        if not counter:
            return None
        
        total = sum(counter.values())
        weights = {}
        for class_id, count in counter.items():
            weights[class_id] = total / (len(counter) * count)
        return weights
    
    def _calculate_lesion_weights(self):
        """Calculate weights for different lesion types"""
        if not self.dataset_stats['lesion_counts']:
            return None
        
        total_lesion_samples = self.dataset_stats['segmentation_samples']
        weights = {}
        for lesion_id, count in self.dataset_stats['lesion_counts'].items():
            weights[lesion_id] = total_lesion_samples / (len(self.dataset_stats['lesion_counts']) * count)
        return weights
    
    def _load_segmentation_masks(self, mask_folder, img_id):
        """Load segmentation masks for all classes"""
        mask_folders = [
            "1. Microaneurysms",
            "2. Haemorrhages", 
            "3. Hard Exudates",
            "4. Soft Exudates",
            "5. Optic Disc"
        ]
        
        mask_suffixes = ["_MA.tif", "_HE.tif", "_EX.tif", "_SE.tif", "_OD.tif"]
        
        masks = []
        for i, (folder, suffix) in enumerate(zip(mask_folders, mask_suffixes)):
            mask_path = os.path.join(mask_folder, folder, f"{img_id}{suffix}")
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Ensure mask has the correct shape
                    mask = cv2.resize(mask, (self.image_size, self.image_size), 
                                    interpolation=cv2.INTER_NEAREST)
                    mask = (mask > 127).astype(np.uint8)
                else:
                    mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            else:
                mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            masks.append(mask)
        
        # Add background mask (inverse of all lesions)
        all_lesions = np.logical_or.reduce(masks)
        background = (~all_lesions).astype(np.uint8)
        masks.insert(0, background)
        
        return np.stack(masks, axis=0)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Prepare outputs
        sample = {
            'image': image,
            'task_type': self.task_types[idx],
            'image_path': img_path
        }
        
        # Handle segmentation data
        if self.segmentation_masks[idx] is not None:
            masks = self.segmentation_masks[idx]
            # Resize masks to match image size
            resized_masks = []
            for i in range(masks.shape[0]):
                mask = cv2.resize(masks[i], (self.image_size, self.image_size), 
                                interpolation=cv2.INTER_NEAREST)
                resized_masks.append(mask)
            masks = np.stack(resized_masks, axis=0)
            sample['seg_mask'] = masks.astype(np.float32)
        else:
            sample['seg_mask'] = np.zeros((self.config.num_seg_classes, self.image_size, self.image_size), dtype=np.float32)
        
        # Handle classification data
        if self.classification_labels[idx] is not None:
            dr_grade, dme_risk = self.classification_labels[idx]
            sample['dr_grade'] = dr_grade
            sample['dme_risk'] = dme_risk
        else:
            sample['dr_grade'] = -1  # Invalid label
            sample['dme_risk'] = -1  # Invalid label
        
        # Apply transforms
        if self.transforms:
            # Apply transforms to image and masks together
            if 'seg_mask' in sample and np.any(sample['seg_mask']):
                # For segmentation samples, apply transforms to both image and mask
                transformed = self.transforms(
                    image=sample['image'],
                    masks=[sample['seg_mask'][i] for i in range(sample['seg_mask'].shape[0])]
                )
                sample['image'] = transformed['image']
                if 'masks' in transformed:
                    sample['seg_mask'] = np.stack(transformed['masks'], axis=0)
            else:
                # For classification-only samples, apply transforms to image only
                transformed = self.transforms(image=sample['image'])
                sample['image'] = transformed['image']
        
        # Convert to tensor if not already done by transforms
        if not isinstance(sample['image'], torch.Tensor):
            sample['image'] = torch.from_numpy(sample['image'].transpose(2, 0, 1)).float() / 255.0
        
        if not isinstance(sample['seg_mask'], torch.Tensor):
            sample['seg_mask'] = torch.from_numpy(sample['seg_mask']).float()
        
        # Convert labels to tensors
        sample['dr_grade'] = torch.tensor(sample['dr_grade'], dtype=torch.long)
        sample['dme_risk'] = torch.tensor(sample['dme_risk'], dtype=torch.long)
        
        return sample

def get_transforms(config, split='train'):
    """Get data transforms for training/validation with advanced augmentation for imbalanced classes"""
    if split == 'train' and config.use_augmentation:
        transforms = [
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=config.rotation_limit, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=config.brightness_limit,
                contrast_limit=config.contrast_limit,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=config.hue_shift_limit,
                sat_shift_limit=config.sat_shift_limit,
                val_shift_limit=0,
                p=0.3
            ),
        ]
        
        # Add advanced augmentation for imbalanced classes
        if config.use_advanced_augmentation:
            # Elastic transform for better lesion detection
            if config.elastic_transform_prob > 0:
                transforms.append(
                    A.ElasticTransform(
                        alpha=1, sigma=50,
                        p=config.elastic_transform_prob
                    )
                )
            
            # Grid distortion for more variety
            transforms.append(
                A.GridDistortion(
                    num_steps=5, distort_limit=0.3,
                    p=0.3
                )
            )
            
            # Optical distortion for realistic variations
            transforms.append(
                A.OpticalDistortion(
                    distort_limit=0.2,
                    p=0.3
                )
            )
            
            # Random gamma for exposure variations
            transforms.append(
                A.RandomGamma(
                    gamma_limit=(80, 120),
                    p=0.3
                )
            )
        
        transforms.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        return A.Compose(transforms)
    else:
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def collate_fn(batch):
    """Custom collate function to handle mixed tasks"""
    images = torch.stack([item['image'] for item in batch])
    seg_masks = torch.stack([item['seg_mask'] for item in batch])
    dr_grades = torch.stack([item['dr_grade'] for item in batch])
    dme_risks = torch.stack([item['dme_risk'] for item in batch])
    task_types = [item['task_type'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'image': images,
        'seg_mask': seg_masks,
        'dr_grade': dr_grades,
        'dme_risk': dme_risks,
        'task_type': task_types,
        'image_path': image_paths
    }

def create_data_loaders(config):
    """Create train and validation data loaders with balanced sampling"""
    train_transforms = get_transforms(config, 'train')
    val_transforms = get_transforms(config, 'val')
    
    # Create datasets with balanced sampling
    train_dataset = IDRiDDataset(config, split='train', transforms=train_transforms, balance_sampling=config.use_balanced_sampling)
    val_dataset = IDRiDDataset(config, split='val', transforms=val_transforms, balance_sampling=False)  # No balancing for validation
    
    # Create sampler for balanced training
    if config.use_balanced_sampling and train_dataset.sample_weights:
        sampler = WeightedRandomSampler(
            weights=train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        shuffle = False  # Sampler handles shuffling
    else:
        sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn
    )
    
    # Log dataset statistics
    logger.info("=" * 60)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    if train_dataset.dataset_stats['dr_grades']:
        logger.info(f"DR Grade distribution (train): {dict(train_dataset.dataset_stats['dr_grades'])}")
    if train_dataset.dataset_stats['dme_risks']:
        logger.info(f"DME Risk distribution (train): {dict(train_dataset.dataset_stats['dme_risks'])}")
    if train_dataset.dataset_stats['lesion_counts']:
        logger.info(f"Lesion distribution (train): {dict(train_dataset.dataset_stats['lesion_counts'])}")
    
    if config.use_balanced_sampling:
        logger.info("Balanced sampling enabled - using WeightedRandomSampler")
        logger.info(f"Sample weight range: {min(train_dataset.sample_weights):.3f} - {max(train_dataset.sample_weights):.3f}")
    else:
        logger.info("Balanced sampling disabled - using standard random sampling")
    
    logger.info("=" * 60)
    
    return train_loader, val_loader, train_dataset.dataset_stats

class AdvancedAugmentation:
    """Advanced augmentation techniques for imbalanced datasets"""
    
    @staticmethod
    def mixup(images, labels, alpha=0.2):
        """Mixup augmentation for classification labels"""
        if alpha <= 0:
            return images, labels
        
        batch_size = images.size(0)
        weights = torch.distributions.Beta(alpha, alpha).sample((batch_size,)).to(images.device)
        index = torch.randperm(batch_size)
        
        mixed_images = weights.view(-1, 1, 1, 1) * images + (1 - weights).view(-1, 1, 1, 1) * images[index]
        
        # For classification labels, we need to handle the mixing differently
        mixed_labels = {}
        for key, label in labels.items():
            if key in ['dr_grade', 'dme_risk']:
                # For classification, we'll use the original labels but with mixed weights
                mixed_labels[key] = label
                mixed_labels[f'{key}_mixup_weight'] = weights
                mixed_labels[f'{key}_mixup_index'] = index
            else:
                mixed_labels[key] = label
        
        return mixed_images, mixed_labels
    
    @staticmethod
    def cutmix(images, labels, prob=0.3):
        """CutMix augmentation for segmentation masks"""
        if prob <= 0 or torch.rand(1) > prob:
            return images, labels
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        # Random cutmix box
        lam = torch.distributions.Beta(1, 1).sample().item()
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(images.size(2) * cut_rat)
        cut_h = int(images.size(3) * cut_rat)
        
        # Random center
        cx = np.random.randint(images.size(2))
        cy = np.random.randint(images.size(3))
        
        bbx1 = np.clip(cx - cut_w // 2, 0, images.size(2))
        bby1 = np.clip(cy - cut_h // 2, 0, images.size(3))
        bbx2 = np.clip(cx + cut_w // 2, 0, images.size(2))
        bby2 = np.clip(cy + cut_h // 2, 0, images.size(3))
        
        # Apply cutmix
        images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
        
        # For segmentation masks, apply the same cutmix
        if 'seg_mask' in labels:
            labels['seg_mask'][:, :, bbx1:bbx2, bby1:bby2] = labels['seg_mask'][index, :, bbx1:bbx2, bby1:bby2]
        
        return images, labels

def apply_advanced_augmentation(batch, config):
    """Apply advanced augmentation techniques to a batch"""
    if not config.use_advanced_augmentation:
        return batch
    
    images = batch['image']
    labels = {
        'dr_grade': batch['dr_grade'],
        'dme_risk': batch['dme_risk'],
        'seg_mask': batch['seg_mask']
    }
    
    # Apply Mixup
    if config.mixup_alpha > 0:
        images, labels = AdvancedAugmentation.mixup(images, labels, config.mixup_alpha)
    
    # Apply CutMix
    if config.cutmix_prob > 0:
        images, labels = AdvancedAugmentation.cutmix(images, labels, config.cutmix_prob)
    
    # Update batch with augmented data
    batch['image'] = images
    batch['dr_grade'] = labels['dr_grade']
    batch['dme_risk'] = labels['dme_risk']
    batch['seg_mask'] = labels['seg_mask']
    
    return batch