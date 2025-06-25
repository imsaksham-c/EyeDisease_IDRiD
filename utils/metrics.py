import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

class MetricsCalculator:
    """Calculate various metrics for evaluation."""
    
    @staticmethod
    def classification_metrics(pred, target):
        """Calculate classification metrics."""
        pred_labels = torch.argmax(pred, dim=1).cpu().numpy()
        target_labels = target.cpu().numpy()
        
        accuracy = accuracy_score(target_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            target_labels, pred_labels, average='weighted', zero_division=0
        )
        
        # AUC (for multi-class, use one-vs-rest)
        try:
            pred_probs = torch.softmax(pred, dim=1).cpu().numpy()
            auc = roc_auc_score(target_labels, pred_probs, multi_class='ovr', average='weighted')
        except:
            auc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    @staticmethod
    def segmentation_metrics(pred, target, threshold=0.5):
        """Calculate segmentation metrics."""
        pred_binary = (pred > threshold).float()
        target_binary = target.float()
        
        # Flatten tensors
        pred_flat = pred_binary.view(-1)
        target_flat = target_binary.view(-1)
        
        # Dice coefficient
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-6)
        
        # IoU
        union = pred_flat.sum() + target_flat.sum() - intersection
        iou = intersection / (union + 1e-6)
        
        # Pixel accuracy
        correct = (pred_flat == target_flat).sum()
        pixel_acc = correct / target_flat.numel()
        
        return {
            'dice': dice.item(),
            'iou': iou.item(),
            'pixel_accuracy': pixel_acc.item()
        }