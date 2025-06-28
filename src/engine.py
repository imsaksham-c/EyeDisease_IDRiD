import torch
import torch.nn as nn
from tqdm import tqdm
from src import config, utils
import cv2

def train_one_epoch(model, dataloader, optimizer, cls_criterion, bce_seg_criterion, dice_loss_fn, device):
    """
    Performs one full training pass over the dataset.
    This version uses a weighted combined BCE + Dice loss for the segmentation task.
    Returns detailed loss breakdowns.
    """
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_seg_loss = 0.0
    all_cls_preds, all_cls_labels = [], []
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch in progress_bar:
        images = batch["image"].to(device, non_blocking=True)
        seg_masks = batch["segmentation_mask"].to(device, non_blocking=True)
        cls_labels = batch["classification_label"].to(device, non_blocking=True)

        optimizer.zero_grad()
        
        outputs = model(images)
        cls_outputs = outputs["classification"]
        seg_outputs = outputs["segmentation"]
        
        # --- Classification Loss (handles missing labels) ---
        cls_valid_mask = cls_labels >= 0
        loss_cls = torch.tensor(0.0, device=device)
        if cls_valid_mask.any():
            loss_cls = cls_criterion(cls_outputs[cls_valid_mask], cls_labels[cls_valid_mask])
            # Track classification accuracy
            cls_preds = torch.argmax(cls_outputs[cls_valid_mask], dim=1)
            all_cls_preds.extend(cls_preds.cpu().numpy())
            all_cls_labels.extend(cls_labels[cls_valid_mask].cpu().numpy())
            
        # --- FIX: Use a weighted sum for the segmentation loss ---
        # This prevents the BCE loss from overwhelming the Dice loss for imbalanced masks.
        loss_bce_seg = bce_seg_criterion(seg_outputs, seg_masks)
        loss_dice_seg = dice_loss_fn(seg_outputs, seg_masks)
        loss_seg = (0.5 * loss_bce_seg) + (0.5 * loss_dice_seg)

        # --- Combined Loss for Backpropagation ---
        combined_loss = loss_seg
        if cls_valid_mask.any():
            combined_loss = (config.LOSS_ALPHA * loss_cls) + ((1 - config.LOSS_ALPHA) * loss_seg)
        
        combined_loss.backward()
        optimizer.step()
        
        total_loss += combined_loss.item()
        total_cls_loss += loss_cls.item()
        total_seg_loss += loss_seg.item()
        
        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))

    # Calculate classification accuracy
    cls_accuracy = 0.0
    if all_cls_labels:
        cls_accuracy = (torch.tensor(all_cls_preds) == torch.tensor(all_cls_labels)).float().mean().item()

    return {
        'total_loss': total_loss / len(dataloader),
        'cls_loss': total_cls_loss / len(dataloader),
        'seg_loss': total_seg_loss / len(dataloader),
        'cls_accuracy': cls_accuracy
    }

@torch.no_grad()
def evaluate(model, dataloader, cls_criterion, bce_seg_criterion, dice_loss_fn, device):
    """
    Evaluates the model on the validation or test set.
    Returns detailed loss breakdowns and metrics.
    """
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_seg_loss = 0.0
    all_cls_preds, all_cls_labels = [], []
    seg_metrics = utils.SegmentationMetrics(num_classes=config.NUM_CLASSES_SEGMENTATION)

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    for batch in progress_bar:
        images = batch["image"].to(device, non_blocking=True)
        seg_masks = batch["segmentation_mask"].to(device, non_blocking=True)
        cls_labels = batch["classification_label"].to(device, non_blocking=True)
        
        outputs = model(images)
        cls_outputs = outputs["classification"]
        seg_outputs = outputs["segmentation"]
        
        # --- Classification Loss & Metrics (handles missing labels) ---
        cls_valid_mask = cls_labels >= 0
        loss_cls = torch.tensor(0.0, device=device)
        if cls_valid_mask.any():
            loss_cls = cls_criterion(cls_outputs[cls_valid_mask], cls_labels[cls_valid_mask])
            cls_preds = torch.argmax(cls_outputs[cls_valid_mask], dim=1)
            all_cls_preds.extend(cls_preds.cpu().numpy())
            all_cls_labels.extend(cls_labels[cls_valid_mask].cpu().numpy())
            
        # --- FIX: Use a weighted sum for the segmentation loss ---
        loss_bce_seg = bce_seg_criterion(seg_outputs, seg_masks)
        loss_dice_seg = dice_loss_fn(seg_outputs, seg_masks)
        loss_seg = (0.5 * loss_bce_seg) + (0.5 * loss_dice_seg)
        
        combined_loss = loss_seg
        if cls_valid_mask.any():
            combined_loss = (config.LOSS_ALPHA * loss_cls) + ((1 - config.LOSS_ALPHA) * loss_seg)
        
        total_loss += combined_loss.item()
        total_cls_loss += loss_cls.item()
        total_seg_loss += loss_seg.item()
        
        # Update segmentation metrics
        seg_preds = torch.sigmoid(seg_outputs) > 0.5
        seg_metrics.update(seg_preds.cpu(), seg_masks.cpu())

    # Calculate classification accuracy
    cls_accuracy = 0.0
    if all_cls_labels:
        cls_accuracy = (torch.tensor(all_cls_preds) == torch.tensor(all_cls_labels)).float().mean().item()
        
    dice_score, iou_score = seg_metrics.get_scores()
    
    return {
        'total_loss': total_loss / len(dataloader),
        'cls_loss': total_cls_loss / len(dataloader),
        'seg_loss': total_seg_loss / len(dataloader),
        'cls_accuracy': cls_accuracy,
        'dice_score': dice_score,
        'iou_score': iou_score
    }
