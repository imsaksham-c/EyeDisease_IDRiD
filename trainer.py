import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from losses import MultiTaskLoss

class Trainer:
    """
    Trainer class for handling training and validation loops for multi-task and single-task models.
    Handles early stopping, learning rate scheduling, and logging.
    """
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
        """
        Runs one training epoch.
        Returns:
            avg_loss (float): Average loss for the epoch
            avg_metrics (dict): Average metrics for the epoch
        """
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
        """
        Runs one validation epoch.
        Returns:
            avg_loss (float): Average loss for the epoch
            avg_metrics (dict): Average metrics for the epoch
        """
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
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(all_targets, all_predictions)
            avg_metrics['classification_accuracy'] = accuracy
        return avg_loss, avg_metrics
    def train(self, num_epochs):
        """
        Runs the full training loop with validation, early stopping, and model saving.
        """
        import logging
        logger = logging.getLogger(__name__)
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