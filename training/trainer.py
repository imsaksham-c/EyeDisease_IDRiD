import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

class Trainer:
    """Training pipeline for multi-task model."""
    
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.config = config
        self.device = config.device
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        # Optimizer and scheduler
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Loss function
        from ..utils.losses import CombinedLoss
        self.criterion = CombinedLoss(
            config.classification_weight,
            config.segmentation_weight
        )
        
        # Metrics
        from ..utils.metrics import MetricsCalculator
        self.metrics_calc = MetricsCalculator()
        
        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            images = batch['image'].to(self.device)
            class_labels = batch['class_label'].to(self.device)
            masks = batch['mask'].to(self.device)
            has_mask = batch['has_mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, task='both')
            
            # Prepare targets
            targets = {
                'class_label': class_labels,
                'mask': masks,
                'has_mask': has_mask
            }
            
            # Compute loss
            loss, loss_dict = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Avg Loss': f"{total_loss / (batch_idx + 1):.4f}"
            })
        
        return total_loss / num_batches
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_cls_preds, all_cls_targets = [], []
        all_seg_preds, all_seg_targets = [], []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                images = batch['image'].to(self.device)
                class_labels = batch['class_label'].to(self.device)
                masks = batch['mask'].to(self.device)
                has_mask = batch['has_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images, task='both')
                
                # Prepare targets
                targets = {
                    'class_label': class_labels,
                    'mask': masks,
                    'has_mask': has_mask
                }
                
                # Compute loss
                loss, _ = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # Collect predictions for metrics
                if 'classification' in outputs:
                    all_cls_preds.append(outputs['classification'].cpu())
                    all_cls_targets.append(class_labels.cpu())
                
                if 'segmentation' in outputs:
                    mask_indices = has_mask > 0
                    if mask_indices.sum() > 0:
                        all_seg_preds.append(outputs['segmentation'][mask_indices].cpu())
                        all_seg_targets.append(masks[mask_indices].cpu())
        
        # Calculate metrics
        metrics = {}
        
        if all_cls_preds:
            cls_preds = torch.cat(all_cls_preds)
            cls_targets = torch.cat(all_cls_targets)
            cls_metrics = self.metrics_calc.classification_metrics(cls_preds, cls_targets)
            metrics.update({f'cls_{k}': v for k, v in cls_metrics.items()})
        
        if all_seg_preds:
            seg_preds = torch.cat(all_seg_preds)
            seg_targets = torch.cat(all_seg_targets)
            seg_metrics = self.metrics_calc.segmentation_metrics(seg_preds, seg_targets)
            metrics.update({f'seg_{k}': v for k, v in seg_metrics.items()})
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss, metrics
    
    def train(self):
        """Complete training loop."""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            for metric_name, value in val_metrics.items():
                print(f"{metric_name}: {value:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                print("New best model saved!")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        filename = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch}.pth'
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)