import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    """Combined loss for multi-task learning."""
    
    def __init__(self, classification_weight=1.0, segmentation_weight=1.0):
        super().__init__()
        self.classification_weight = classification_weight
        self.segmentation_weight = segmentation_weight
        
        self.classification_loss = nn.CrossEntropyLoss()
        self.segmentation_loss = nn.BCELoss()
    
    def forward(self, outputs, targets):
        total_loss = 0
        loss_dict = {}
        
        if 'classification' in outputs and 'class_label' in targets:
            cls_loss = self.classification_loss(outputs['classification'], targets['class_label'])
            total_loss += self.classification_weight * cls_loss
            loss_dict['classification_loss'] = cls_loss.item()
        
        if 'segmentation' in outputs and 'mask' in targets:
            # Only compute segmentation loss for samples with masks
            has_mask = targets.get('has_mask', torch.ones(len(targets['mask'])))
            if has_mask.sum() > 0:
                mask_indices = has_mask > 0
                seg_pred = outputs['segmentation'][mask_indices].squeeze(1)
                seg_target = targets['mask'][mask_indices]
                seg_loss = self.segmentation_loss(seg_pred, seg_target)
                total_loss += self.segmentation_weight * seg_loss
                loss_dict['segmentation_loss'] = seg_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict

class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice
