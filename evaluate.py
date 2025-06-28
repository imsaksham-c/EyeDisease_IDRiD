import torch
import torch.nn as nn
import argparse
import os
from src import config, utils, data_loader, model, engine

def run(args):
    """Main function to run the evaluation on the test set using a trained model."""
    utils.seed_everything(config.SEED)
    
    # --- Get Test Data Loader ---
    _, test_loader = data_loader.get_loaders()
    print("Test data loaded.")

    # --- Load Trained Model ---
    device = config.DEVICE
    eval_model = model.get_model().to(device)
    eval_model = utils.load_model(eval_model, args.model_path, device)
    print("Model loaded for evaluation.")

    # --- Define Loss Functions ---
    cls_criterion = nn.CrossEntropyLoss()
    seg_criterion_bce = nn.BCEWithLogitsLoss()
    seg_criterion_dice = utils.DiceLoss().to(device)

    # --- Run Evaluation ---
    print("Evaluating on the test set...")
    test_metrics = engine.evaluate(eval_model, test_loader, cls_criterion, seg_criterion_bce, seg_criterion_dice, device)

    print("\n--- Evaluation on Test Set Finished ---")
    print(f"  Test Total Loss: {test_metrics['total_loss']:.4f}")
    print(f"  Test Classification Loss: {test_metrics['cls_loss']:.4f}")
    print(f"  Test Segmentation Loss: {test_metrics['seg_loss']:.4f}")
    print(f"  Test Classification Accuracy: {test_metrics['cls_accuracy']:.4f}")
    print(f"  Test Segmentation Dice Score: {test_metrics['dice_score']:.4f}")
    print(f"  Test Segmentation IoU Score:  {test_metrics['iou_score']:.4f}")
    
    # --- Save Visual Results ---
    print("\nSaving visual results from the test set...")
    utils.save_visual_results(eval_model, test_loader, device, num_samples=15)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the multi-task model on the test set.")
    parser.add_argument(
        '--model_path', 
        type=str, 
        default=os.path.join(config.MODEL_PATH, "best_model.pth"),
        help='Path to the trained model file (.pth). Defaults to the best model saved during training.'
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Please run train.py first or provide a valid path using --model_path.")
    else:
        run(args)
