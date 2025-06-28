import torch
import cv2
import numpy as np
import argparse
import os
from src import config, utils, model
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from datetime import datetime
import json

def predict(image_path, model_path, device):
    """
    Loads a model and runs prediction on a single image.
    Saves the output visualization and individual segmentation masks.
    """
    # --- Load Model ---
    print("Loading model...")
    prediction_model = model.get_model().to(device)
    prediction_model = utils.load_model(prediction_model, model_path, device)
    prediction_model.eval()
    print("Model loaded successfully.")

    # --- Preprocess Image ---
    print(f"Loading and preprocessing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply the same transformations as the validation set
    val_transforms = A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    transformed = val_transforms(image=image)
    input_tensor = transformed['image'].unsqueeze(0).to(device) # Add batch dimension

    # --- Run Prediction ---
    print("Running inference...")
    with torch.no_grad():
        outputs = prediction_model(input_tensor)

    # --- Process Outputs ---
    # Classification
    cls_output = outputs["classification"]
    cls_prob = torch.softmax(cls_output, dim=1)
    predicted_grade = torch.argmax(cls_prob, dim=1).item()
    confidence = cls_prob.max().item()
    all_probabilities = cls_prob.cpu().numpy()[0].tolist()
    
    # Segmentation
    seg_output = outputs["segmentation"]
    pred_mask = (torch.sigmoid(seg_output) > 0.5).cpu().squeeze(0) # Remove batch dimension

    print(f"Predicted DR Grade: {predicted_grade} (Confidence: {confidence:.2f})")

    # --- Create output directory with datetime ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("runs", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Save original image ---
    print("Saving original image...")
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    original_image_resized = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    original_image_bgr = cv2.cvtColor(original_image_resized, cv2.COLOR_RGB2BGR)
    original_path = os.path.join(output_dir, f"{base_filename}_original.png")
    cv2.imwrite(original_path, original_image_bgr)
    print(f"Saved original image: {original_path}")
    
    # --- Save individual segmentation masks ---
    print("Saving individual segmentation masks...")
    class_names = ["MA", "HE", "EX", "SE", "OD"]  # Microaneurysms, Haemorrhages, Hard Exudates, Soft Exudates, Optic Disc
    
    # Convert pred_mask to numpy if it's a tensor
    if isinstance(pred_mask, torch.Tensor):
        pred_mask_np = pred_mask.detach().cpu().numpy()
    else:
        pred_mask_np = pred_mask
    
    for i, class_name in enumerate(class_names):
        if i < pred_mask_np.shape[0]:  # Ensure we don't exceed available channels
            # Extract single channel mask
            single_mask = pred_mask_np[i, :, :].astype(np.uint8) * 255  # Convert to 0-255 range
            
            # Save as TIFF
            mask_filename = f"{base_filename}_{class_name}.tif"
            mask_path = os.path.join(output_dir, mask_filename)
            cv2.imwrite(mask_path, single_mask)
            print(f"Saved {class_name} mask: {mask_path}")

    # --- Save classification result as JSON ---
    print("Saving classification result as JSON...")
    classification_result = {
        "image": os.path.basename(image_path),
        "predicted_grade": predicted_grade,
        "confidence": confidence,
        "probabilities": all_probabilities
    }
    json_path = os.path.join(output_dir, f"{base_filename}_classification.json")
    with open(json_path, 'w') as f:
        json.dump(classification_result, f, indent=4)
    print(f"Saved classification result: {json_path}")

    # --- Save Visualization ---
    # (Removed stacked prediction and matplotlib visualization)
    print(f"All outputs saved in: {output_dir}")


def predict_classification_only(image_path, model_path, device):
    """
    Runs only classification prediction on a single image.
    Returns the predicted grade and confidence.
    """
    # --- Load Model ---
    print("Loading model for classification...")
    prediction_model = model.get_model().to(device)
    prediction_model = utils.load_model(prediction_model, model_path, device)
    prediction_model.eval()
    print("Model loaded successfully.")

    # --- Preprocess Image ---
    print(f"Loading and preprocessing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return None, None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply the same transformations as the validation set
    val_transforms = A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    transformed = val_transforms(image=image)
    input_tensor = transformed['image'].unsqueeze(0).to(device) # Add batch dimension

    # --- Run Prediction ---
    print("Running classification inference...")
    with torch.no_grad():
        outputs = prediction_model(input_tensor)

    # --- Process Classification Outputs ---
    cls_output = outputs["classification"]
    cls_prob = torch.softmax(cls_output, dim=1)
    predicted_grade = torch.argmax(cls_prob, dim=1).item()
    confidence = cls_prob.max().item()
    
    # Get all class probabilities
    all_probabilities = cls_prob.cpu().numpy()[0]
    
    print(f"Predicted DR Grade: {predicted_grade} (Confidence: {confidence:.2f})")
    print("All class probabilities:")
    for i, prob in enumerate(all_probabilities):
        print(f"  Grade {i}: {prob:.3f}")
    
    return predicted_grade, confidence, all_probabilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction on a single fundus image.")
    parser.add_argument(
        '--image', 
        type=str, 
        required=True, 
        help='Path to the input image file.'
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
        default=os.path.join(config.MODEL_PATH, "best_model.pth"),
        help='Path to the trained model file (.pth). Defaults to the best model saved during training.'
    )
    parser.add_argument(
        '--classification_only',
        action='store_true',
        help='Run only classification prediction (no segmentation).'
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Input image not found at {args.image}")
    elif not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Please run train.py first or provide a valid path using --model_path.")
    else:
        if args.classification_only:
            predict_classification_only(args.image, args.model_path, config.DEVICE)
        else:
            predict(args.image, args.model_path, config.DEVICE)
