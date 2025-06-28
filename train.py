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
from utils.losses import MultiTaskLoss
from utils.data_utils import EyeDiseaseDataset, create_data_transforms
from utils.dataset_reorganiser import DatasetReorganizer

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
                images = batch['image'].to(self.device)
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
    dataset_reorganizer = DatasetReorganizer(DATASET_PATH)
    reorganized_data = dataset_reorganizer.reorganize_data()
    
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