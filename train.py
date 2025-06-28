import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
from src import config, utils, data_loader, model, engine

def run():
    """Main function to orchestrate the model training and validation pipeline."""
    utils.seed_everything(config.SEED)
    
    os.makedirs(config.MODEL_PATH, exist_ok=True)
    os.makedirs(config.RESULTS_PATH, exist_ok=True)

    print("Loading data...")
    train_loader, val_loader = data_loader.get_loaders()
    print("Data loaded successfully.")

    print("Initializing model...")
    device = config.DEVICE
    mt_model = model.get_model().to(device)
    
    optimizer = optim.Adam(mt_model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, verbose=True)
    
    # --- Loss Functions ---
    # Standard loss for multi-class classification
    cls_criterion = nn.CrossEntropyLoss()
    
    # Define both BCE and Dice loss for the segmentation task
    # Standard pixel-wise loss
    seg_criterion_bce = nn.BCEWithLogitsLoss()
    # Loss for handling class imbalance
    seg_criterion_dice = utils.DiceLoss().to(device)
    
    print(f"Model initialized on device: {device}")

    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'cls_accuracy': [], 'dice_score': [], 'iou_score': []}

    print(f"--- Starting Training for {config.EPOCHS} Epochs ---")
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        
        # --- FIX: Pass both segmentation losses to the engine functions ---
        train_metrics = engine.train_one_epoch(
            mt_model, train_loader, optimizer, cls_criterion, seg_criterion_bce, seg_criterion_dice, device
        )
        val_metrics = engine.evaluate(
            mt_model, val_loader, cls_criterion, seg_criterion_bce, seg_criterion_dice, device
        )
        
        scheduler.step(val_metrics['total_loss'])

        # --- Detailed reporting with loss breakdowns ---
        print(f"  Train Total Loss: {train_metrics['total_loss']:.4f}")
        print(f"  Train Classification Loss: {train_metrics['cls_loss']:.4f}")
        print(f"  Train Segmentation Loss: {train_metrics['seg_loss']:.4f}")
        print(f"  Train Classification Accuracy: {train_metrics['cls_accuracy']:.4f}")
        print(f"  Val Total Loss: {val_metrics['total_loss']:.4f}")
        print(f"  Val Classification Loss: {val_metrics['cls_loss']:.4f}")
        print(f"  Val Segmentation Loss: {val_metrics['seg_loss']:.4f}")
        print(f"  Val Classification Accuracy: {val_metrics['cls_accuracy']:.4f}")
        print(f"  Val Segmentation Dice: {val_metrics['dice_score']:.4f}")
        print(f"  Val Segmentation IoU: {val_metrics['iou_score']:.4f}")
        
        # Best metrics so far
        print(f"  Best Val Loss so far: {min(history['val_loss']+[val_metrics['total_loss']]):.4f}")
        print(f"  Best Val Dice so far: {max(history['dice_score']+[val_metrics['dice_score']]):.4f}")
        print(f"  Best Val IoU so far: {max(history['iou_score']+[val_metrics['iou_score']]):.4f}")
        print(f"  Best Val Accuracy so far: {max(history['cls_accuracy']+[val_metrics['cls_accuracy']]):.4f}")

        history['train_loss'].append(train_metrics['total_loss'])
        history['val_loss'].append(val_metrics['total_loss'])
        history['cls_accuracy'].append(val_metrics['cls_accuracy'])
        history['dice_score'].append(val_metrics['dice_score'])
        history['iou_score'].append(val_metrics['iou_score'])

        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            epochs_no_improve = 0
            utils.save_model(mt_model, optimizer, epoch, os.path.join(config.MODEL_PATH, "best_model.pth"))
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

    print("\n--- Training Finished ---")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(config.OUTPUT_PATH, "training_history.csv"), index=False)
    print(f"Training history saved to {os.path.join(config.OUTPUT_PATH, 'training_history.csv')}")

if __name__ == "__main__":
    run()
