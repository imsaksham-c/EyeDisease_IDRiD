import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining classification and segmentation losses.
    Allows for weighted sum of both losses.
    """
    def __init__(self, classification_weight=1.0, segmentation_weight=1.0):
        super(MultiTaskLoss, self).__init__()
        self.classification_weight = classification_weight
        self.segmentation_weight = segmentation_weight
        self.classification_loss = nn.CrossEntropyLoss()
        self.segmentation_loss = nn.BCELoss()
    
    def forward(self, predictions, targets):
        total_loss = 0
        loss_dict = {}
        # Classification loss
        if 'classification' in predictions and 'retinopathy_grade' in targets:
            cls_loss = self.classification_loss(predictions['classification'], targets['retinopathy_grade'])
            total_loss += self.classification_weight * cls_loss
            loss_dict['classification_loss'] = cls_loss.item()
        # Segmentation loss (only for samples with masks)
        if 'segmentation' in predictions and 'segmentation_mask' in targets:
            valid_mask = targets['has_segmentation']
            if valid_mask.sum() > 0:
                valid_pred = predictions['segmentation'][valid_mask]
                valid_target = targets['segmentation_mask'][valid_mask]
                # Resize predictions to match target size if needed
                if valid_pred.size() != valid_target.size():
                    valid_pred = F.interpolate(valid_pred, size=valid_target.shape[-2:], mode='bilinear', align_corners=False)
                seg_loss = self.segmentation_loss(valid_pred, valid_target)
                total_loss += self.segmentation_weight * seg_loss
                loss_dict['segmentation_loss'] = seg_loss.item()
        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict 