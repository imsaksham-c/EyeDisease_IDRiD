# Fixed training script with proper gating logging
import os
import sys
import logging
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
from datetime import datetime

# Add src to path
sys.path.append('src')

from config import Config
from dataset import create_data_loaders, apply_advanced_augmentation
from model import MultiTaskModel
from loss import CombinedLoss, MetricsCalculator

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def setup_logging(config):
    """Setup logging configuration"""
    os.makedirs(config.log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_task_masks(batch):
    """Create task masks indicating which samples have valid targets"""
    task_masks = {}
    
    # Classification mask - samples with valid DR or DME labels
    has_cls = (batch['dr_grade'] >= 0) | (batch['dme_risk'] >= 0)
    task_masks['has_classification'] = has_cls
    
    # Segmentation mask - samples with valid segmentation masks
    has_seg = batch['seg_mask'].sum(dim=(1, 2, 3)) > 0
    task_masks['has_segmentation'] = has_seg
    
    return task_masks

def train_epoch(model, data_loader, criterion, optimizer, scaler, config, logger):
    """Train for one epoch with advanced augmentation"""
    model.train()
    
    running_loss = 0.0
    all_losses = {}
    all_metrics = {}
    
    progress_bar = tqdm(data_loader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        images = batch['image'].to(config.device)
        dr_grades = batch['dr_grade'].to(config.device)
        dme_risks = batch['dme_risk'].to(config.device)
        seg_masks = batch['seg_mask'].to(config.device)
        
        # Apply advanced augmentation for imbalanced classes
        if config.use_advanced_augmentation:
            batch['image'] = images
            batch['dr_grade'] = dr_grades
            batch['dme_risk'] = dme_risks
            batch['seg_mask'] = seg_masks
            batch = apply_advanced_augmentation(batch, config)
            images = batch['image']
            dr_grades = batch['dr_grade']
            dme_risks = batch['dme_risk']
            seg_masks = batch['seg_mask']
        
        # Create task masks
        task_masks = create_task_masks(batch)
        for key in task_masks:
            task_masks[key] = task_masks[key].to(config.device)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast(enabled=config.mixed_precision):
            # Model forward pass
            predictions = model(images)
            
            # Prepare targets
            targets = {
                'dr_grade': dr_grades,
                'dme_risk': dme_risks,
                'seg_mask': seg_masks
            }
            
            # Calculate loss
            loss, loss_dict = criterion(predictions, targets, task_masks)
        
        # Backward pass
        if config.mixed_precision:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Update running statistics
        running_loss += loss.item()
        
        # Accumulate loss statistics
        for key, value in loss_dict.items():
            if key not in all_losses:
                all_losses[key] = []
            all_losses[key].append(value)
        
        # Calculate metrics for valid samples
        with torch.no_grad():
            # Classification metrics
            cls_mask = task_masks['has_classification']
            if cls_mask.any():
                dr_valid = dr_grades[cls_mask & (dr_grades >= 0)]
                dme_valid = dme_risks[cls_mask & (dme_risks >= 0)]
                
                if len(dr_valid) > 0:
                    dr_preds = predictions['cls_output']['dr_logits'][cls_mask & (dr_grades >= 0)]
                    dr_acc = MetricsCalculator.calculate_accuracy(dr_preds, dr_valid)
                    if 'dr_accuracy' not in all_metrics:
                        all_metrics['dr_accuracy'] = []
                    all_metrics['dr_accuracy'].append(dr_acc)
                
                if len(dme_valid) > 0:
                    dme_preds = predictions['cls_output']['dme_logits'][cls_mask & (dme_risks >= 0)]
                    dme_acc = MetricsCalculator.calculate_accuracy(dme_preds, dme_valid)
                    if 'dme_accuracy' not in all_metrics:
                        all_metrics['dme_accuracy'] = []
                    all_metrics['dme_accuracy'].append(dme_acc)
            
            # Segmentation metrics
            seg_mask = task_masks['has_segmentation']
            if seg_mask.any():
                seg_preds = torch.sigmoid(predictions['seg_output'][seg_mask])
                seg_targets = seg_masks[seg_mask]
                
                # Calculate mean Dice score across all classes
                dice_scores = []
                for i in range(seg_preds.size(1)):
                    dice = MetricsCalculator.calculate_dice_score(
                        seg_preds[:, i], seg_targets[:, i]
                    )
                    dice_scores.append(dice)
                
                mean_dice = np.mean(dice_scores)
                if 'mean_dice' not in all_metrics:
                    all_metrics['mean_dice'] = []
                all_metrics['mean_dice'].append(mean_dice)
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'Cls_Gate': f"{loss_dict.get('avg_cls_gate', 0):.3f}",
            'Seg_Gate': f"{loss_dict.get('avg_seg_gate', 0):.3f}"
        })
        
        # Log to wandb if enabled
        if config.use_wandb and batch_idx % config.log_every_n_steps == 0:
            wandb_dict = {f"train/{k}": v for k, v in loss_dict.items()}
            wandb_dict['train/step'] = batch_idx
            wandb.log(wandb_dict)
    
    # Calculate epoch averages
    epoch_loss = running_loss / len(data_loader)
    epoch_losses = {k: np.mean(v) for k, v in all_losses.items()}
    epoch_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    
    # Log gating statistics
    logger.info(f"Gating Stats - Cls Gate: {epoch_losses.get('avg_cls_gate', 0):.3f} ± {epoch_losses.get('cls_gate_std', 0):.3f}")
    logger.info(f"Gating Stats - Seg Gate: {epoch_losses.get('avg_seg_gate', 0):.3f} ± {epoch_losses.get('seg_gate_std', 0):.3f}")
    
    return epoch_loss, epoch_losses, epoch_metrics

def validate_epoch(model, data_loader, criterion, config, logger):
    """Validate for one epoch"""
    model.eval()
    
    running_loss = 0.0
    all_losses = {}
    all_metrics = {}
    
    progress_bar = tqdm(data_loader, desc="Validation")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch['image'].to(config.device)
            dr_grades = batch['dr_grade'].to(config.device)
            dme_risks = batch['dme_risk'].to(config.device)
            seg_masks = batch['seg_mask'].to(config.device)
            
            # Create task masks
            task_masks = create_task_masks(batch)
            for key in task_masks:
                task_masks[key] = task_masks[key].to(config.device)
            
            # Forward pass
            predictions = model(images)
            
            # Prepare targets
            targets = {
                'dr_grade': dr_grades,
                'dme_risk': dme_risks,
                'seg_mask': seg_masks
            }
            
            # Calculate loss
            loss, loss_dict = criterion(predictions, targets, task_masks)
            
            running_loss += loss.item()
            
            # Accumulate loss statistics
            for key, value in loss_dict.items():
                if key not in all_losses:
                    all_losses[key] = []
                all_losses[key].append(value)
            
            # Calculate metrics
            # Classification metrics
            cls_mask = task_masks['has_classification']
            if cls_mask.any():
                dr_valid = dr_grades[cls_mask & (dr_grades >= 0)]
                dme_valid = dme_risks[cls_mask & (dme_risks >= 0)]
                
                if len(dr_valid) > 0:
                    dr_preds = predictions['cls_output']['dr_logits'][cls_mask & (dr_grades >= 0)]
                    dr_acc = MetricsCalculator.calculate_accuracy(dr_preds, dr_valid)
                    if 'dr_accuracy' not in all_metrics:
                        all_metrics['dr_accuracy'] = []
                    all_metrics['dr_accuracy'].append(dr_acc)
                
                if len(dme_valid) > 0:
                    dme_preds = predictions['cls_output']['dme_logits'][cls_mask & (dme_risks >= 0)]
                    dme_acc = MetricsCalculator.calculate_accuracy(dme_preds, dme_valid)
                    if 'dme_accuracy' not in all_metrics:
                        all_metrics['dme_accuracy'] = []
                    all_metrics['dme_accuracy'].append(dme_acc)
            
            # Segmentation metrics
            seg_mask = task_masks['has_segmentation']
            if seg_mask.any():
                seg_preds = torch.sigmoid(predictions['seg_output'][seg_mask])
                seg_targets = seg_masks[seg_mask]
                
                # Calculate mean Dice score
                dice_scores = []
                for i in range(seg_preds.size(1)):
                    dice = MetricsCalculator.calculate_dice_score(
                        seg_preds[:, i], seg_targets[:, i]
                    )
                    dice_scores.append(dice)
                
                mean_dice = np.mean(dice_scores)
                if 'mean_dice' not in all_metrics:
                    all_metrics['mean_dice'] = []
                all_metrics['mean_dice'].append(mean_dice)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Cls_Gate': f"{loss_dict.get('avg_cls_gate', 0):.3f}",
                'Seg_Gate': f"{loss_dict.get('avg_seg_gate', 0):.3f}"
            })
    
    # Calculate epoch averages
    epoch_loss = running_loss / len(data_loader)
    epoch_losses = {k: np.mean(v) for k, v in all_losses.items()}
    epoch_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    
    return epoch_loss, epoch_losses, epoch_metrics

def save_checkpoint(model, optimizer, epoch, loss, config, filename):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config.__dict__
    }
    
    checkpoint_path = os.path.join(config.checkpoint_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path

def main(args):
    # Initialize configuration
    config = Config()
    
    # Override config with command line arguments
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.no_wandb:
        config.use_wandb = False
    if args.no_balanced_sampling:
        config.use_balanced_sampling = False
    if args.no_class_weights:
        config.use_class_weights = False
    
    # Set random seeds
    set_random_seeds(config.random_seed)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info(f"Starting training with config: {config}")
    
    # Create directories
    config.create_directories()
    
    # Initialize wandb if enabled
    if config.use_wandb:
        wandb.init(
            project="multitask-eye-disease",
            config=vars(config),
            name=f"balanced_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Create data loaders with balanced sampling
    logger.info("Creating data loaders...")
    train_loader, val_loader, dataset_stats = create_data_loaders(config)
    
    # Initialize model
    logger.info("Initializing model...")
    model = MultiTaskModel(config).to(config.device)
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Initialize loss function with class weights
    class_weights = dataset_stats['class_weights'] if config.use_class_weights else None
    criterion = CombinedLoss(config, class_weights=class_weights)
    
    # Initialize mixed precision training
    scaler = GradScaler() if config.mixed_precision else None
    
    # Training loop
    logger.info("Starting training loop...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        logger.info(f"\n=== Epoch {epoch + 1}/{config.num_epochs} ===")
        
        # Train
        train_loss, train_losses, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, config, logger
        )
        
        # Validate
        val_loss, val_losses, val_metrics = validate_epoch(
            model, val_loader, criterion, config, logger
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log epoch results
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        
        if 'dr_accuracy' in train_metrics:
            logger.info(f"Train DR Accuracy: {train_metrics['dr_accuracy']:.4f}")
        if 'dme_accuracy' in train_metrics:
            logger.info(f"Train DME Accuracy: {train_metrics['dme_accuracy']:.4f}")
        if 'mean_dice' in train_metrics:
            logger.info(f"Train Mean Dice: {train_metrics['mean_dice']:.4f}")
        
        # Log gating statistics
        if 'avg_cls_gate' in val_losses:
            logger.info(f"Val Gating - Cls: {val_losses['avg_cls_gate']:.3f}, Seg: {val_losses['avg_seg_gate']:.3f}")
        
        # Log to wandb
        if config.use_wandb:
            wandb_dict = {
                'epoch': epoch,
                'train/loss': train_loss,
                'val/loss': val_loss,
                'train/lr': optimizer.param_groups[0]['lr']
            }
            
            # Add detailed losses
            for key, value in train_losses.items():
                wandb_dict[f'train/{key}'] = value
            for key, value in val_losses.items():
                wandb_dict[f'val/{key}'] = value
            
            # Add metrics
            for key, value in train_metrics.items():
                wandb_dict[f'train/{key}'] = value
            for key, value in val_metrics.items():
                wandb_dict[f'val/{key}'] = value
            
            wandb.log(wandb_dict)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss, config, 'best_model.pth')
            logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if not config.save_best_only:
                save_checkpoint(model, optimizer, epoch, val_loss, config, f'checkpoint_epoch_{epoch}.pth')
        
        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            logger.info(f"Early stopping triggered after {config.early_stopping_patience} epochs without improvement")
            break
    
    logger.info("Training completed!")
    if config.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Task Eye Disease Model")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--no_balanced_sampling", action="store_true", help="Disable balanced sampling")
    parser.add_argument("--no_class_weights", action="store_true", help="Disable class weights in loss")
    
    args = parser.parse_args()
    main(args)