"""
train.py
--------
Main script to train multi-task and single-task models on the IDRiD dataset. Handles data loading, model initialization, training, and evaluation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import pandas as pd
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import logging
from tqdm import tqdm
import json
import random
from utils.models import ExpertModule, GatingNetwork, ModularMultiTaskModel, SingleTaskModel
from utils.trainer import Trainer
from utils.evaluator import Evaluator
from utils.data_utils import EyeDiseaseDataset
from utils.dataset_reorganiser import reorganized_data

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# This file has been renamed to train.py. All DatasetReorganizer logic is now in dataset_reorganiser.py.

class EyeDiseaseDataset(Dataset):
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
        combined_mask = np.zeros((target_size[0], target_size[1]), dtype=np.float32)
        
        for mask_type, mask_path in mask_paths.items():
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = cv2.resize(mask, (target_size[1], target_size[0]))
                    mask = (mask > 127).astype(np.float32)
                    combined_mask = np.maximum(combined_mask, mask)
        
        return torch.from_numpy(combined_mask).unsqueeze(0)

class ExpertModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super(ExpertModule, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_experts),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.gate(x)

class ModularMultiTaskModel(nn.Module):
    def __init__(self, num_classes_classification=5, num_experts=3, hidden_dim=512):
        super(ModularMultiTaskModel, self).__init__()
        
        # Shared backbone
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone_features = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Feature dimension after backbone
        self.feature_dim = self.backbone_features
        
        # Expert modules for classification
        self.classification_experts = nn.ModuleList([
            ExpertModule(self.feature_dim, hidden_dim, num_classes_classification)
            for _ in range(num_experts)
        ])
        
        # Expert modules for segmentation
        self.segmentation_experts = nn.ModuleList([
            self._create_segmentation_expert()
            for _ in range(num_experts)
        ])
        
        # Gating networks
        self.classification_gate = GatingNetwork(self.feature_dim, num_experts)
        self.segmentation_gate = GatingNetwork(self.feature_dim, num_experts)
        
        # Task selector (learns to route between tasks)
        self.task_selector = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 2),
            nn.Sigmoid()
        )
    
    def _create_segmentation_expert(self):
        return nn.Sequential(
            nn.ConvTranspose2d(self.feature_dim, 512, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, task_type='both'):
        batch_size = x.size(0)
        
        # Extract features using shared backbone
        features = self.backbone(x)
        features_flat = features.view(batch_size, -1)
        
        # Task routing weights
        task_weights = self.task_selector(features_flat)
        
        outputs = {}
        
        if task_type in ['both', 'classification']:
            # Classification path
            cls_gate_weights = self.classification_gate(features_flat)
            cls_expert_outputs = []
            
            for expert in self.classification_experts:
                expert_out = expert(features_flat)
                cls_expert_outputs.append(expert_out)
            
            cls_expert_outputs = torch.stack(cls_expert_outputs, dim=2)
            cls_gate_weights = cls_gate_weights.unsqueeze(1)
            
            classification_output = torch.sum(cls_expert_outputs * cls_gate_weights, dim=2)
            outputs['classification'] = classification_output
            outputs['cls_gate_weights'] = cls_gate_weights.squeeze()
        
        if task_type in ['both', 'segmentation']:
            # Segmentation path
            seg_gate_weights = self.segmentation_gate(features_flat)
            seg_expert_outputs = []
            
            # Reshape features for segmentation experts
            features_2d = features.view(batch_size, self.feature_dim, 1, 1)
            
            for expert in self.segmentation_experts:
                expert_out = expert(features_2d)
                seg_expert_outputs.append(expert_out)
            
            seg_expert_outputs = torch.stack(seg_expert_outputs, dim=1)
            seg_gate_weights = seg_gate_weights.view(batch_size, -1, 1, 1, 1)
            
            segmentation_output = torch.sum(seg_expert_outputs * seg_gate_weights, dim=1)
            outputs['segmentation'] = segmentation_output
            outputs['seg_gate_weights'] = seg_gate_weights.squeeze()
        
        outputs['task_weights'] = task_weights
        return outputs

class MultiTaskLoss(nn.Module):
    def __init__(self, classification_weight=1.0, segmentation_weight=1.0):
        super(MultiTaskLoss, self).__init__()
        self.classification_weight = classification_weight
        self.segmentation_weight = segmentation_weight
        self.classification_loss = nn.CrossEntropyLoss()
        self.segmentation_loss = nn.BCELoss()
    
    def forward(self, predictions, targets):
        total_loss = 0
        loss_dict = {}
        
        if 'classification' in predictions and 'retinopathy_grade' in targets:
            cls_loss = self.classification_loss(predictions['classification'], targets['retinopathy_grade'])
            total_loss += self.classification_weight * cls_loss
            loss_dict['classification_loss'] = cls_loss.item()
        
        if 'segmentation' in predictions and 'segmentation_mask' in targets:
            # Filter out samples without segmentation masks
            valid_mask = targets['has_segmentation']
            if valid_mask.sum() > 0:
                valid_pred = predictions['segmentation'][valid_mask]
                valid_target = targets['segmentation_mask'][valid_mask]
                
                # Resize predictions to match target size
                if valid_pred.size() != valid_target.size():
                    valid_pred = F.interpolate(valid_pred, size=valid_target.shape[-2:], mode='bilinear', align_corners=False)
                
                seg_loss = self.segmentation_loss(valid_pred, valid_target)
                total_loss += self.segmentation_weight * seg_loss
                loss_dict['segmentation_loss'] = seg_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, lr=1e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)
        self.criterion = MultiTaskLoss()
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.early_stopping_patience = 10
        self.early_stopping_counter = 0
    
    def train_epoch(self):
        self.model.train()
        epoch_losses = []
        epoch_metrics = {'classification_loss': [], 'segmentation_loss': [], 'total_loss': []}
        
        for batch in tqdm(self.train_loader, desc="Training"):
            self.optimizer.zero_grad()
            
            images = batch['image'].to(self.device)
            targets = {
                'retinopathy_grade': batch['retinopathy_grade'].to(self.device),
                'has_segmentation': batch['has_segmentation'].to(self.device)
            }
            
            if batch['segmentation_mask'][0] is not None:
                seg_masks = []
                for mask in batch['segmentation_mask']:
                    if mask is not None:
                        seg_masks.append(mask)
                    else:
                        seg_masks.append(torch.zeros(1, 224, 224))
                targets['segmentation_mask'] = torch.stack(seg_masks).to(self.device)
            
            outputs = self.model(images, task_type='both')
            loss, loss_dict = self.criterion(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
            for key, value in loss_dict.items():
                if key in epoch_metrics:
                    epoch_metrics[key].append(value)
        
        avg_loss = np.mean(epoch_losses)
        avg_metrics = {k: np.mean(v) if v else 0 for k, v in epoch_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def validate_epoch(self):
        self.model.eval()
        epoch_losses = []
        epoch_metrics = {'classification_loss': [], 'segmentation_loss': [], 'total_loss': []}
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                targets = {
                    'retinopathy_grade': batch['retinopathy_grade'].to(self.device),
                    'has_segmentation': batch['has_segmentation'].to(self.device)
                }
                
                if batch['segmentation_mask'][0] is not None:
                    seg_masks = []
                    for mask in batch['segmentation_mask']:
                        if mask is not None:
                            seg_masks.append(mask)
                        else:
                            seg_masks.append(torch.zeros(1, 224, 224))
                    targets['segmentation_mask'] = torch.stack(seg_masks).to(self.device)
                
                outputs = self.model(images, task_type='both')
                loss, loss_dict = self.criterion(outputs, targets)
                
                epoch_losses.append(loss.item())
                for key, value in loss_dict.items():
                    if key in epoch_metrics:
                        epoch_metrics[key].append(value)
                
                # Collect predictions for metrics
                if 'classification' in outputs:
                    pred_classes = torch.argmax(outputs['classification'], dim=1)
                    all_predictions.extend(pred_classes.cpu().numpy())
                    all_targets.extend(targets['retinopathy_grade'].cpu().numpy())
        
        avg_loss = np.mean(epoch_losses)
        avg_metrics = {k: np.mean(v) if v else 0 for k, v in epoch_metrics.items()}
        
        # Calculate classification accuracy
        if all_predictions and all_targets:
            accuracy = accuracy_score(all_targets, all_predictions)
            avg_metrics['classification_accuracy'] = accuracy
        
        return avg_loss, avg_metrics
    
    def train(self, num_epochs):
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_metrics = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch()
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Logging
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            logger.info(f"Train Metrics: {train_metrics}")
            logger.info(f"Val Metrics: {val_metrics}")
            
            # Early stopping and model saving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
                logger.info("New best model saved!")
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.early_stopping_patience:
                    logger.info("Early stopping triggered!")
                    break
        
        logger.info("Training completed!")

class SingleTaskModel(nn.Module):
    def __init__(self, num_classes=5, task_type='classification'):
        super(SingleTaskModel, self).__init__()
        self.task_type = task_type
        
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone_features = self.backbone.fc.in_features
        
        if task_type == 'classification':
            self.backbone.fc = nn.Linear(self.backbone_features, num_classes)
        elif task_type == 'segmentation':
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            self.segmentation_head = nn.Sequential(
                nn.ConvTranspose2d(self.backbone_features, 512, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, 4, 2, 1),
                nn.Sigmoid()
            )
    
    def forward(self, x):
        if self.task_type == 'classification':
            return {'classification': self.backbone(x)}
        elif self.task_type == 'segmentation':
            features = self.backbone(x)
            seg_output = self.segmentation_head(features)
            return {'segmentation': seg_output}

class Evaluator:
    def __init__(self, model, test_loader, device):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
    
    def evaluate(self):
        self.model.eval()
        all_predictions = []
        all_targets = []
        segmentation_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                images = batch['image'].to(device)
                targets = batch['retinopathy_grade'].cpu().numpy()
                
                outputs = self.model(images, task_type='both')
                
                if 'classification' in outputs:
                    pred_classes = torch.argmax(outputs['classification'], dim=1)
                    all_predictions.extend(pred_classes.cpu().numpy())
                    all_targets.extend(targets)
                
                if 'segmentation' in outputs and batch['segmentation_mask'][0] is not None:
                    seg_pred = outputs['segmentation']
                    seg_masks = []
                    for mask in batch['segmentation_mask']:
                        if mask is not None:
                            seg_masks.append(mask)
                        else:
                            seg_masks.append(torch.zeros(1, 224, 224))
                    seg_target = torch.stack(seg_masks).to(self.device)
                    
                    # Calculate IoU for segmentation
                    if seg_target.sum() > 0:
                        seg_pred_binary = (seg_pred > 0.5).float()
                        intersection = (seg_pred_binary * seg_target).sum()
                        union = seg_pred_binary.sum() + seg_target.sum() - intersection
                        iou = intersection / (union + 1e-8)
                        segmentation_metrics.append(iou.item())
        
        # Classification metrics
        accuracy = accuracy_score(all_targets, all_predictions) if all_targets else 0
        
        # Segmentation metrics
        avg_iou = np.mean(segmentation_metrics) if segmentation_metrics else 0
        
        return {
            'classification_accuracy': accuracy,
            'segmentation_iou': avg_iou,
            'confusion_matrix': confusion_matrix(all_targets, all_predictions) if all_targets else None
        }

def create_data_transforms():
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

def visualize_results(model, test_loader, device, num_samples=5):
    model.eval()
    samples_shown = 0
    
    plt.figure(figsize=(15, 3 * num_samples))
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if samples_shown >= num_samples:
                break
            
            images = batch['image'].to(device)
            outputs = model(images, task_type='both')
            
            for i in range(min(images.size(0), num_samples - samples_shown)):
                # Original image
                img = images[i].cpu()
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                
                plt.subplot(num_samples, 3, samples_shown * 3 + 1)
                plt.imshow(img.permute(1, 2, 0))
                plt.title(f"Original Image {samples_shown + 1}")
                plt.axis('off')
                
                # Classification prediction
                if 'classification' in outputs:
                    pred_class = torch.argmax(outputs['classification'][i]).item()
                    actual_class = batch['retinopathy_grade'][i].item()
                    plt.subplot(num_samples, 3, samples_shown * 3 + 2)
                    plt.text(0.5, 0.5, f"Predicted: {pred_class}\nActual: {actual_class}", 
                            ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
                    plt.title("Classification")
                    plt.axis('off')
                
                # Segmentation prediction
                if 'segmentation' in outputs:
                    seg_pred = outputs['segmentation'][i, 0].cpu().numpy()
                    plt.subplot(num_samples, 3, samples_shown * 3 + 3)
                    plt.imshow(seg_pred, cmap='hot')
                    plt.title("Segmentation Prediction")
                    plt.axis('off')
                
                samples_shown += 1
                if samples_shown >= num_samples:
                    break
    
    plt.tight_layout()
    plt.savefig('results_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def dice_coefficient(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def calculate_metrics(predictions, targets, task_type='classification'):
    if task_type == 'classification':
        accuracy = accuracy_score(targets, predictions)
        cm = confusion_matrix(targets, predictions)
        return {'accuracy': accuracy, 'confusion_matrix': cm}
    elif task_type == 'segmentation':
        dice_scores = []
        iou_scores = []
        
        for pred, target in zip(predictions, targets):
            # Dice coefficient
            dice = dice_coefficient(pred, target)
            dice_scores.append(dice.item())
            
            # IoU
            pred_binary = (pred > 0.5).float()
            target_binary = (target > 0.5).float()
            intersection = (pred_binary * target_binary).sum()
            union = pred_binary.sum() + target_binary.sum() - intersection
            iou = intersection / (union + 1e-8)
            iou_scores.append(iou.item())
        
        return {
            'dice_score': np.mean(dice_scores),
            'iou_score': np.mean(iou_scores),
            'dice_std': np.std(dice_scores),
            'iou_std': np.std(iou_scores)
        }

def generate_report(multitask_results, single_task_results, training_history):
    report = []
    report.append("# Modular Vision-Based Multi-task Learning for Eye Disease Diagnosis")
    report.append("## Performance Analysis Report")
    report.append("")
    
    # Model Architecture
    report.append("### Model Architecture")
    report.append("- **Backbone**: ResNet-50 pretrained on ImageNet")
    report.append("- **Expert Modules**: 3 specialized processing units for each task")
    report.append("- **Dynamic Routing**: Gating networks for expert selection")
    report.append("- **Tasks**: Disease grading classification + Lesion segmentation")
    report.append("")
    
    # Performance Metrics
    report.append("### Performance Metrics")
    report.append("")
    report.append("#### Multi-task Model Performance:")
    report.append(f"- Classification Accuracy: {multitask_results['classification_accuracy']:.4f}")
    report.append(f"- Segmentation IoU: {multitask_results['segmentation_iou']:.4f}")
    report.append("")
    
    report.append("#### Single-task Baseline Performance:")
    report.append(f"- Classification-only Accuracy: {single_task_results.get('classification_accuracy', 'N/A')}")
    report.append(f"- Segmentation-only IoU: {single_task_results.get('segmentation_iou', 'N/A')}")
    report.append("")
    
    # Analysis
    report.append("### Analysis")
    if multitask_results['classification_accuracy'] > single_task_results.get('classification_accuracy', 0):
        report.append("- **Classification**: Multi-task model outperforms single-task baseline")
        report.append("  - Benefit from shared representations and joint learning")
    else:
        report.append("- **Classification**: Single-task model shows superior performance")
        report.append("  - Task interference may be affecting multi-task learning")
    
    if multitask_results['segmentation_iou'] > single_task_results.get('segmentation_iou', 0):
        report.append("- **Segmentation**: Multi-task model shows better performance")
        report.append("  - Classification features help segmentation task")
    else:
        report.append("- **Segmentation**: Single-task model performs better")
        report.append("  - Dedicated resources improve segmentation accuracy")
    
    report.append("")
    report.append("### Training Details")
    report.append(f"- Total Epochs: {len(training_history['train_losses'])}")
    report.append(f"- Best Validation Loss: {min(training_history['val_losses']):.4f}")
    report.append(f"- Final Training Loss: {training_history['train_losses'][-1]:.4f}")
    report.append("")
    
    # Recommendations
    report.append("### Recommendations")
    report.append("1. **Data Augmentation**: Increase dataset diversity for better generalization")
    report.append("2. **Loss Balancing**: Fine-tune task-specific loss weights")
    report.append("3. **Architecture Tuning**: Experiment with different numbers of experts")
    report.append("4. **Hyperparameter Optimization**: Systematic search for optimal parameters")
    report.append("")
    
    return "\n".join(report)

def save_confusion_matrix(cm, class_names, filename='confusion_matrix.png'):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Disease Grading Classification')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def create_expert_analysis_plot(model, test_loader, device):
    model.eval()
    gate_weights_cls = []
    gate_weights_seg = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            outputs = model(images, task_type='both')
            
            if 'cls_gate_weights' in outputs:
                gate_weights_cls.extend(outputs['cls_gate_weights'].cpu().numpy())
            if 'seg_gate_weights' in outputs:
                gate_weights_seg.extend(outputs['seg_gate_weights'].cpu().numpy())
    
    # Plot expert utilization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    if gate_weights_cls:
        gate_weights_cls = np.array(gate_weights_cls)
        ax1.boxplot([gate_weights_cls[:, i] for i in range(gate_weights_cls.shape[1])],
                   labels=[f'Expert {i+1}' for i in range(gate_weights_cls.shape[1])])
        ax1.set_title('Classification Expert Utilization')
        ax1.set_ylabel('Gate Weight')
    
    if gate_weights_seg:
        gate_weights_seg = np.array(gate_weights_seg)
        ax2.boxplot([gate_weights_seg[:, i] for i in range(gate_weights_seg.shape[1])],
                   labels=[f'Expert {i+1}' for i in range(gate_weights_seg.shape[1])])
        ax2.set_title('Segmentation Expert Utilization')
        ax2.set_ylabel('Gate Weight')
    
    plt.tight_layout()
    plt.savefig('expert_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

class ModelProfiler:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
    
    def profile_inference_time(self, input_size=(1, 3, 224, 224), num_runs=100):
        dummy_input = torch.randn(input_size).to(self.device)
        self.model.eval()
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        # Timing
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(num_runs):
            start_time.record()
            with torch.no_grad():
                _ = self.model(dummy_input)
            end_time.record()
            torch.cuda.synchronize()
            times.append(start_time.elapsed_time(end_time))
        
        return {
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times)
        }
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }

def advanced_evaluation(model, test_loader, device):
    model.eval()
    
    all_cls_predictions = []
    all_cls_targets = []
    all_seg_predictions = []
    all_seg_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Advanced Evaluation"):
            images = batch['image'].to(device)
            outputs = model(images, task_type='both')
            
            # Classification metrics
            if 'classification' in outputs:
                cls_pred = torch.argmax(outputs['classification'], dim=1)
                all_cls_predictions.extend(cls_pred.cpu().numpy())
                all_cls_targets.extend(batch['retinopathy_grade'].numpy())
            
            # Segmentation metrics
            if 'segmentation' in outputs and batch['segmentation_mask'][0] is not None:
                seg_pred = outputs['segmentation']
                
                for i, mask in enumerate(batch['segmentation_mask']):
                    if mask is not None:
                        seg_target = mask.to(device)
                        if seg_pred.size() != seg_target.size():
                            seg_pred_resized = F.interpolate(
                                seg_pred[i:i+1], 
                                size=seg_target.shape[-2:], 
                                mode='bilinear', 
                                align_corners=False
                            )
                        else:
                            seg_pred_resized = seg_pred[i:i+1]
                        
                        all_seg_predictions.append(seg_pred_resized.cpu())
                        all_seg_targets.append(seg_target.cpu())
    
    # Calculate detailed metrics
    results = {}
    
    if all_cls_predictions:
        cls_metrics = calculate_metrics(all_cls_predictions, all_cls_targets, 'classification')
        results.update(cls_metrics)
    
    if all_seg_predictions:
        seg_metrics = calculate_metrics(all_seg_predictions, all_seg_targets, 'segmentation')
        results.update(seg_metrics)
    
    return results

def save_training_history(trainer, filename='training_history.json'):
    history = {
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'best_val_loss': trainer.best_val_loss
    }
    
    with open(filename, 'w') as f:
        json.dump(history, f, indent=2)

def plot_training_history(trainer):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(trainer.train_losses, label='Training Loss')
    plt.plot(trainer.val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    epochs = range(1, len(trainer.train_losses) + 1)
    plt.plot(epochs, trainer.train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, trainer.val_losses, 'ro-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_models(multitask_results, single_task_results):
    print("\n" + "="*50)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*50)
    
    print(f"Multi-task Model:")
    print(f"  Classification Accuracy: {multitask_results['classification_accuracy']:.4f}")
    print(f"  Segmentation IoU: {multitask_results['segmentation_iou']:.4f}")
    
    print(f"\nSingle-task Classification Model:")
    print(f"  Classification Accuracy: {single_task_results.get('classification_accuracy', 'N/A')}")
    
    print(f"\nSingle-task Segmentation Model:")
    print(f"  Segmentation IoU: {single_task_results.get('segmentation_iou', 'N/A')}")
    
    # Analysis
    if multitask_results['classification_accuracy'] > single_task_results.get('classification_accuracy', 0):
        print(f"\nMulti-task model shows BETTER classification performance")
    else:
        print(f"\nSingle-task model shows BETTER classification performance")
    
    if multitask_results['segmentation_iou'] > single_task_results.get('segmentation_iou', 0):
        print(f"Multi-task model shows BETTER segmentation performance")
    else:
        print(f"Single-task model shows BETTER segmentation performance")

def custom_collate_fn(batch):
    # Assume all images are the same size as the first image
    image_shape = batch[0]['image'].shape if batch[0]['image'] is not None else (3, 224, 224)
    for item in batch:
        if item['segmentation_mask'] is None:
            item['segmentation_mask'] = torch.zeros(1, image_shape[1], image_shape[2])
    return torch.utils.data.dataloader.default_collate(batch)

def main():
    # Configuration
    DATASET_PATH = "dataset"
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {DEVICE}")
    
    # Step 1: Reorganize dataset
    logger.info("Reorganizing dataset...")
    # The DatasetReorganizer logic is now in dataset_reorganiser.py
    
    # Step 2: Create data transforms
    train_transform, val_transform = create_data_transforms()
    
    # Step 3: Create datasets and data loaders
    train_dataset = EyeDiseaseDataset(
        reorganized_data['train'], 
        transform=train_transform, 
        task_type='both'
    )
    
    # Split training data for validation (80-20 split)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    test_dataset = EyeDiseaseDataset(
        reorganized_data['test'], 
        transform=val_transform, 
        task_type='both'
    )
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    
    logger.info(f"Dataset sizes - Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_dataset)}")
    
    # Step 4: Initialize and train multi-task model
    logger.info("Initializing multi-task model...")
    multitask_model = ModularMultiTaskModel(num_classes_classification=5, num_experts=3)
    
    trainer = Trainer(multitask_model, train_loader, val_loader, DEVICE, lr=LEARNING_RATE)
    trainer.train(NUM_EPOCHS)
    
    # Step 5: Evaluate multi-task model
    logger.info("Evaluating multi-task model...")
    multitask_model.load_state_dict(torch.load('best_model.pth'))
    evaluator = Evaluator(multitask_model, test_loader, DEVICE)
    multitask_results = evaluator.evaluate()
    
    # Step 6: Train and evaluate single-task baseline models
    logger.info("Training single-task classification model...")
    single_task_cls_model = SingleTaskModel(num_classes=5, task_type='classification')
    cls_trainer = Trainer(single_task_cls_model, train_loader, val_loader, DEVICE, lr=LEARNING_RATE)
    cls_trainer.train(NUM_EPOCHS // 2)  # Train for fewer epochs for comparison
    
    single_task_cls_model.load_state_dict(torch.load('best_model.pth'))
    cls_evaluator = Evaluator(single_task_cls_model, test_loader, DEVICE)
    single_task_cls_results = cls_evaluator.evaluate()
    
    logger.info("Training single-task segmentation model...")
    # Create segmentation-only dataset
    seg_train_dataset = EyeDiseaseDataset(
        [item for item in reorganized_data['train'] if item['has_segmentation']], 
        transform=train_transform, 
        task_type='segmentation'
    )
    seg_test_dataset = EyeDiseaseDataset(
        [item for item in reorganized_data['test'] if item['has_segmentation']], 
        transform=val_transform, 
        task_type='segmentation'
    )
    
    if len(seg_train_dataset) > 0:
        seg_train_size = int(0.8 * len(seg_train_dataset))
        seg_val_size = len(seg_train_dataset) - seg_train_size
        seg_train_subset, seg_val_subset = torch.utils.data.random_split(
            seg_train_dataset, [seg_train_size, seg_val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        seg_train_loader = DataLoader(seg_train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
        seg_val_loader = DataLoader(seg_val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
        seg_test_loader = DataLoader(seg_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
        
        single_task_seg_model = SingleTaskModel(task_type='segmentation')
        seg_trainer = Trainer(single_task_seg_model, seg_train_loader, seg_val_loader, DEVICE, lr=LEARNING_RATE)
        seg_trainer.train(NUM_EPOCHS // 2)
        
        single_task_seg_model.load_state_dict(torch.load('best_model.pth'))
        seg_evaluator = Evaluator(single_task_seg_model, seg_test_loader, DEVICE)
        single_task_seg_results = seg_evaluator.evaluate()
        
        single_task_results = {
            'classification_accuracy': single_task_cls_results['classification_accuracy'],
            'segmentation_iou': single_task_seg_results['segmentation_iou']
        }
    else:
        single_task_results = {
            'classification_accuracy': single_task_cls_results['classification_accuracy'],
            'segmentation_iou': 0.0
        }
    
    # Step 7: Compare models and generate report
    logger.info("Generating comparison report...")
    compare_models(multitask_results, single_task_results)
    
    # Step 8: Visualize results
    logger.info("Generating visualizations...")
    visualize_results(multitask_model, test_loader, DEVICE, num_samples=5)
    plot_training_history(trainer)
    save_training_history(trainer)
    
    # Step 9: Save final results
    final_results = {
        'multitask_model': multitask_results,
        'single_task_model': single_task_results,
        'training_history': {
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses
        }
    }
    
    with open('final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    logger.info("Training and evaluation completed successfully!")
    logger.info(f"Results saved to: final_results.json")
    logger.info(f"Best model saved to: best_model.pth")
    logger.info(f"Visualizations saved to: results_visualization.png, training_history.png")
    
    return final_results

if __name__ == "__main__":
    results = main() 