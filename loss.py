# Loss functions for multi-task learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        """
        logits: [B, C, H, W]
        targets: [B, C, H, W]
        """
        # Apply sigmoid to logits
        probs = torch.sigmoid(logits)
        
        # Flatten tensors
        probs = probs.view(probs.size(0), probs.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)
        
        # Calculate Dice score for each class
        intersection = (probs * targets).sum(dim=2)
        union = probs.sum(dim=2) + targets.sum(dim=2)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedLoss(nn.Module):
    """Combined loss for multi-task learning with gating and class weights"""
    def __init__(self, config, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.config = config
        self.class_weights = class_weights or {}
        
        # Classification losses with class weights
        if self.class_weights.get('dr_weights'):
            dr_weights = torch.tensor([self.class_weights['dr_weights'].get(i, 1.0) for i in range(config.num_classes_dr)])
            self.dr_ce_loss = nn.CrossEntropyLoss(weight=dr_weights, ignore_index=-1)
        else:
            self.dr_ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        
        if self.class_weights.get('dme_weights'):
            dme_weights = torch.tensor([self.class_weights['dme_weights'].get(i, 1.0) for i in range(config.num_classes_dme)])
            self.dme_ce_loss = nn.CrossEntropyLoss(weight=dme_weights, ignore_index=-1)
        else:
            self.dme_ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        
        self.focal_loss = FocalLoss(alpha=1, gamma=2)
        
        # Segmentation losses with lesion weights
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        
        # Uncertainty-based weighting parameters
        self.log_vars = nn.Parameter(torch.zeros(2))  # [cls_var, seg_var]
        
        # Log class weights if available
        if self.class_weights:
            logger.info("Class weights loaded for loss calculation:")
            if self.class_weights.get('dr_weights'):
                logger.info(f"DR weights: {self.class_weights['dr_weights']}")
            if self.class_weights.get('dme_weights'):
                logger.info(f"DME weights: {self.class_weights['dme_weights']}")
            if self.class_weights.get('lesion_weights'):
                logger.info(f"Lesion weights: {self.class_weights['lesion_weights']}")
        
    def forward(self, predictions, targets, task_masks=None):
        """
        Compute combined loss with gating and class weights
        
        Args:
            predictions: Model predictions dict
            targets: Target dict with 'dr_grade', 'dme_risk', 'seg_mask'
            task_masks: Dict indicating which samples have valid targets
        """
        device = predictions['gates'].device
        batch_size = predictions['gates'].size(0)
        
        # Initialize losses
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}
        
        # Get gating weights
        gates = predictions['gates']  # [B, 2] - [cls_gate, seg_gate]
        cls_gates = gates[:, 0]  # Classification gates
        seg_gates = gates[:, 1]  # Segmentation gates
        
        # Classification loss with class weights
        cls_loss = torch.tensor(0.0, device=device)
        if 'dr_grade' in targets and 'dme_risk' in targets:
            # DR grading loss
            dr_targets = targets['dr_grade']
            valid_dr_mask = (dr_targets >= 0)
            
            if valid_dr_mask.any():
                dr_logits = predictions['cls_output']['dr_logits'][valid_dr_mask]
                dr_targets_valid = dr_targets[valid_dr_mask]
                dr_loss = self.dr_ce_loss(dr_logits, dr_targets_valid)
                cls_loss += dr_loss
                loss_dict['dr_loss'] = dr_loss.item()
            
            # DME risk loss
            dme_targets = targets['dme_risk']
            valid_dme_mask = (dme_targets >= 0)
            
            if valid_dme_mask.any():
                dme_logits = predictions['cls_output']['dme_logits'][valid_dme_mask]
                dme_targets_valid = dme_targets[valid_dme_mask]
                dme_loss = self.dme_ce_loss(dme_logits, dme_targets_valid)
                cls_loss += dme_loss
                loss_dict['dme_loss'] = dme_loss.item()
        
        # Segmentation loss with lesion weights
        seg_loss = torch.tensor(0.0, device=device)
        if 'seg_mask' in targets:
            seg_logits = predictions['seg_output']
            seg_targets = targets['seg_mask']
            
            # Check for valid segmentation targets
            valid_seg_mask = (seg_targets.sum(dim=(1, 2, 3)) > 0)
            
            if valid_seg_mask.any():
                seg_logits_valid = seg_logits[valid_seg_mask]
                seg_targets_valid = seg_targets[valid_seg_mask]
                
                # Ensure same spatial dimensions
                if seg_logits_valid.shape[-2:] != seg_targets_valid.shape[-2:]:
                    seg_logits_valid = F.interpolate(
                        seg_logits_valid,
                        size=seg_targets_valid.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Apply lesion weights if available
                if self.class_weights.get('lesion_weights'):
                    lesion_weights = torch.ones(seg_logits_valid.size(1), device=device)
                    for i, weight in self.class_weights['lesion_weights'].items():
                        if i < len(lesion_weights):
                            lesion_weights[i] = weight
                    
                    # Weighted BCE loss
                    bce_loss = F.binary_cross_entropy_with_logits(
                        seg_logits_valid, seg_targets_valid, 
                        reduction='none'
                    )
                    # Apply weights per class
                    weighted_bce = (bce_loss * lesion_weights.view(1, -1, 1, 1)).mean()
                else:
                    weighted_bce = self.bce_loss(seg_logits_valid, seg_targets_valid)
                
                # Dice loss (already handles class imbalance well)
                dice_loss = self.dice_loss(seg_logits_valid, seg_targets_valid)
                
                seg_loss = (self.config.bce_loss_weight * weighted_bce + 
                          self.config.dice_loss_weight * dice_loss)
                
                loss_dict['bce_loss'] = weighted_bce.item()
                loss_dict['dice_loss'] = dice_loss.item()
                loss_dict['seg_loss'] = seg_loss.item()
        
        # Gating supervision loss
        gating_loss = torch.tensor(0.0, device=device)
        if task_masks is not None:
            # Create target gates based on available tasks
            target_gates = torch.zeros_like(gates)
            
            # Set classification gate to 1 for classification samples
            if 'has_classification' in task_masks:
                cls_mask = task_masks['has_classification']
                target_gates[cls_mask, 0] = 1.0
            
            # Set segmentation gate to 1 for segmentation samples  
            if 'has_segmentation' in task_masks:
                seg_mask = task_masks['has_segmentation']
                target_gates[seg_mask, 1] = 1.0
            
            # Normalize target gates
            target_gates = F.softmax(target_gates, dim=1)
            
            # KL divergence loss for gating
            gating_loss = F.kl_div(
                F.log_softmax(gates, dim=1),
                target_gates,
                reduction='batchmean'
            )
            loss_dict['gating_loss'] = gating_loss.item()
        
        # Uncertainty-based weighting
        if self.config.cls_loss_weight > 0 and cls_loss > 0:
            cls_precision = torch.exp(-self.log_vars[0])
            weighted_cls_loss = cls_precision * cls_loss + self.log_vars[0]
            total_loss += weighted_cls_loss
            loss_dict['weighted_cls_loss'] = weighted_cls_loss.item()
        
        if self.config.seg_loss_weight > 0 and seg_loss > 0:
            seg_precision = torch.exp(-self.log_vars[1])
            weighted_seg_loss = seg_precision * seg_loss + self.log_vars[1]
            total_loss += weighted_seg_loss
            loss_dict['weighted_seg_loss'] = weighted_seg_loss.item()
        
        # Add gating loss
        if self.config.gating_loss_weight > 0:
            total_loss += self.config.gating_loss_weight * gating_loss
        
        loss_dict['total_loss'] = total_loss.item()
        loss_dict['cls_loss'] = cls_loss.item()
        loss_dict['seg_loss'] = seg_loss.item()
        
        # Store gate statistics
        loss_dict['avg_cls_gate'] = cls_gates.mean().item()
        loss_dict['avg_seg_gate'] = seg_gates.mean().item()
        loss_dict['cls_gate_std'] = cls_gates.std().item()
        loss_dict['seg_gate_std'] = seg_gates.std().item()
        
        return total_loss, loss_dict

class MetricsCalculator:
    """Calculate various metrics for evaluation"""
    
    @staticmethod
    def calculate_dice_score(pred_mask, true_mask, threshold=0.5):
        """Calculate Dice score for segmentation"""
        pred_binary = (pred_mask > threshold).float()
        intersection = (pred_binary * true_mask).sum()
        union = pred_binary.sum() + true_mask.sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        dice = (2.0 * intersection) / union
        return dice.item()
    
    @staticmethod
    def calculate_iou(pred_mask, true_mask, threshold=0.5):
        """Calculate IoU for segmentation"""
        pred_binary = (pred_mask > threshold).float()
        intersection = (pred_binary * true_mask).sum()
        union = pred_binary.sum() + true_mask.sum() - intersection
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        iou = intersection / union
        return iou.item()
    
    @staticmethod
    def calculate_accuracy(predictions, targets):
        """Calculate classification accuracy"""
        pred_classes = torch.argmax(predictions, dim=1)
        correct = (pred_classes == targets).float()
        accuracy = correct.mean()
        return accuracy.item()
    
    @staticmethod
    def calculate_quadratic_kappa(predictions, targets, num_classes):
        """Calculate quadratic weighted kappa for ordinal classification"""
        pred_classes = torch.argmax(predictions, dim=1)
        
        # Convert to numpy for sklearn compatibility
        y_true = targets.cpu().numpy()
        y_pred = pred_classes.cpu().numpy()
        
        # Create weight matrix
        weights = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                weights[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
        
        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
        
        # Calculate kappa
        n = len(y_true)
        hist_true = np.bincount(y_true, minlength=num_classes)
        hist_pred = np.bincount(y_pred, minlength=num_classes)
        
        E = np.outer(hist_true, hist_pred).astype(float) / n
        
        numerator = np.sum(weights * cm)
        denominator = np.sum(weights * E)
        
        if denominator == 0:
            return 0.0
        
        kappa = 1 - (numerator / denominator)
        return kappa