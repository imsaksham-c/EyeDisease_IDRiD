import torch
import torch.nn as nn
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from src import config

def seed_everything(seed=config.SEED):
    """Sets random seeds for reproducibility across all libraries."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_model(model, optimizer, epoch, file_path):
    """Saves the model's state dictionary and optimizer state."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, file_path)
    print(f"Model saved to {file_path}")

def load_model(model, file_path, device):
    """Loads a model's state dictionary from a file."""
    checkpoint = torch.load(file_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {file_path}")
    return model

class DiceLoss(nn.Module):
    """
    Implements Dice Loss for segmentation tasks.
    Dice Loss is effective for imbalanced datasets as it focuses on overlap.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Apply sigmoid to the raw model output (logits) to get probabilities
        inputs = torch.sigmoid(inputs)
        
        # Flatten both the inputs and targets to a 1D tensor
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate the intersection and the total number of elements
        intersection = (inputs * targets).sum()
        total = inputs.sum() + targets.sum()
        
        # Calculate the Dice coefficient
        dice_coeff = (2. * intersection + self.smooth) / (total + self.smooth)
        
        # Return the Dice Loss (1 - Dice Coefficient)
        return 1 - dice_coeff

class SegmentationMetrics:
    """Helper class to compute segmentation metrics (Dice and IoU)."""
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.intersection = torch.zeros(num_classes)
        self.total_preds = torch.zeros(num_classes)
        self.total_targets = torch.zeros(num_classes)
        self.smooth = 1e-6

    def update(self, preds, targets):
        """Update metrics with a new batch of predictions and targets."""
        preds = preds.float()
        targets = targets.float()
        
        for i in range(self.num_classes):
            pred_i = preds[:, i, :, :]
            target_i = targets[:, i, :, :]
            
            intersection = (pred_i * target_i).sum()
            self.intersection[i] += intersection
            self.total_preds[i] += pred_i.sum()
            self.total_targets[i] += target_i.sum()

    def get_scores(self):
        """Calculate the average Dice and IoU scores across all classes."""
        dice = (2. * self.intersection + self.smooth) / (self.total_preds + self.total_targets + self.smooth)
        iou = (self.intersection + self.smooth) / (self.total_preds + self.total_targets - self.intersection + self.smooth)
        return dice.mean().item(), iou.mean().item()


def save_visual_results(model, dataloader, device, num_samples=10):
    """Saves visual examples of segmentation predictions."""
    model.eval()
    os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            if count >= num_samples: break
            
            images = batch["image"].to(device)
            true_masks = batch["segmentation_mask"]
            cls_labels = batch["classification_label"]
            
            outputs = model(images)
            pred_masks = (torch.sigmoid(outputs["segmentation"]) > 0.5).cpu()
            cls_preds = torch.argmax(outputs["classification"], dim=1).cpu()

            for i in range(images.size(0)):
                if count >= num_samples: break
                
                # Un-normalize image for visualization
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)

                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                fig.suptitle(f"True Grade: {cls_labels[i].item()}, Pred Grade: {cls_preds[i].item()}", fontsize=16)
                
                axes[0].imshow(img)
                axes[0].set_title("Original Image")
                axes[0].axis("off")
                
                axes[1].imshow(img)
                axes[1].imshow(create_overlay(true_masks[i]), alpha=0.6)
                axes[1].set_title("Ground Truth Mask")
                axes[1].axis("off")

                axes[2].imshow(img)
                axes[2].imshow(create_overlay(pred_masks[i]), alpha=0.6)
                axes[2].set_title("Predicted Mask")
                axes[2].axis("off")
                
                plt.tight_layout()
                plt.savefig(os.path.join(config.RESULTS_PATH, f"result_sample_{count}.png"))
                plt.close(fig)
                
                count += 1
    print(f"Saved {count} visual results to {config.RESULTS_PATH}")

def create_overlay(mask):
    """Creates a colored overlay from a multi-channel mask for visualization."""
    if not isinstance(mask, np.ndarray):
        mask = mask.detach().cpu().numpy()  # Convert from torch tensor if needed
    colors = np.array([
        [1, 0, 0],  # Red for MA
        [0, 1, 0],  # Green for HE
        [1, 1, 0],  # Yellow for EX
        [0, 0, 1],  # Blue for SE
        [0, 1, 1]   # Cyan for OD
    ], dtype=np.float32)
    
    # Handle different mask shapes
    if len(mask.shape) == 3:
        # Multi-channel mask: (channels, height, width)
        overlay = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.float32)
        num_channels = min(mask.shape[0], len(colors))
        
        for i in range(num_channels):
            channel_mask = mask[i, :, :]
            # Broadcast the channel mask to 3D for color multiplication
            channel_mask_3d = channel_mask[:, :, np.newaxis]
            overlay += channel_mask_3d * colors[i]
    else:
        # Single channel mask: (height, width)
        overlay = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
        mask_3d = mask[:, :, np.newaxis]
        overlay += mask_3d * colors[0]  # Use first color for single channel
        
    return np.clip(overlay, 0, 1)
