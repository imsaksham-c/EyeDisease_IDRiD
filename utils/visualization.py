import matplotlib.pyplot as plt
import numpy as np
import torch
import os

class Visualizer:
    """Visualization utilities."""
    
    @staticmethod
    def plot_training_curves(train_losses, val_losses, save_path=None):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def visualize_predictions(images, predictions, targets, save_path=None, max_samples=8):
        """Visualize model predictions."""
        n_samples = min(len(images), max_samples)
        fig, axes = plt.subplots(3, n_samples, figsize=(3*n_samples, 9))
        
        for i in range(n_samples):
            # Original image
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            img = np.clip(img, 0, 1)
            
            axes[0, i].imshow(img)
            axes[0, i].set_title('Original Image')
            axes[0, i].axis('off')
            
            # Ground truth mask
            if targets is not None:
                gt_mask = targets[i].cpu().numpy()
                axes[1, i].imshow(gt_mask, cmap='gray')
                axes[1, i].set_title('Ground Truth')
            else:
                axes[1, i].text(0.5, 0.5, 'No GT', ha='center', va='center')
            axes[1, i].axis('off')
            
            # Predicted mask
            pred_mask = predictions[i].cpu().numpy()
            if len(pred_mask.shape) == 3:
                pred_mask = pred_mask[0]  # Remove channel dimension
            axes[2, i].imshow(pred_mask, cmap='gray')
            axes[2, i].set_title('Prediction')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
        """Plot confusion matrix."""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()