"""
predict.py
----------
Script for running inference on a single image from the IDRiD dataset using a trained model.
"""

import torch
from utils.models import ModularMultiTaskModel
from torchvision import transforms
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# Configuration
MODEL_PATH = "best_model.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define test transforms (should match validation transforms used in training)
test_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

def create_runs_folder():
    """Create runs folder with timestamp for organizing predictions."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = f"runs/prediction_{timestamp}"
    os.makedirs(runs_dir, exist_ok=True)
    
    # Create subdirectories for different outputs
    seg_dir = os.path.join(runs_dir, "segmentation_masks")
    os.makedirs(seg_dir, exist_ok=True)
    
    # Create subdirectories for each lesion type
    lesion_types = ['Microaneurysms', 'Haemorrhages', 'Hard_Exudates', 'Soft_Exudates', 'Optic_Disc']
    for lesion_type in lesion_types:
        os.makedirs(os.path.join(seg_dir, lesion_type), exist_ok=True)
    
    return runs_dir, seg_dir

def save_segmentation_mask(mask, image_id, lesion_type, output_dir):
    """Save segmentation mask as an image file."""
    # Convert mask to PIL Image and save
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    filename = f"{image_id}_{lesion_type}.png"
    filepath = os.path.join(output_dir, lesion_type, filename)
    mask_img.save(filepath)
    return filepath

def save_combined_segmentation(seg_preds, image_id, output_dir):
    """Save combined segmentation visualization."""
    lesion_types = ['Microaneurysms', 'Haemorrhages', 'Hard_Exudates', 'Soft_Exudates', 'Optic_Disc']
    
    # Create a combined visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (mask, lesion_type) in enumerate(zip(seg_preds, lesion_types)):
        if i < 5:  # We have 5 lesion types
            axes[i].imshow(mask, cmap='hot', alpha=0.7)
            axes[i].set_title(f'{lesion_type}')
            axes[i].axis('off')
    
    # Hide the last subplot
    axes[5].axis('off')
    
    plt.suptitle(f'Segmentation Predictions - {image_id}', fontsize=16)
    plt.tight_layout()
    
    # Save the combined visualization
    combined_filename = f"{image_id}_combined_segmentation.png"
    combined_filepath = os.path.join(output_dir, combined_filename)
    plt.savefig(combined_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return combined_filepath

def find_image_in_dataset(image_name):
    """Find an image in the dataset by name."""
    # Search in both segmentation and disease grading datasets
    search_paths = [
        os.path.join("dataset", "A. Segmentation", "1. Original Images", "a. Training Set"),
        os.path.join("dataset", "A. Segmentation", "1. Original Images", "b. Testing Set"),
        os.path.join("dataset", "B. Disease Grading", "1. Original Images", "a. Training Set"),
        os.path.join("dataset", "B. Disease Grading", "1. Original Images", "b. Testing Set")
    ]
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            # Try different extensions
            for ext in ['.jpg', '.jpeg', '.png']:
                image_path = os.path.join(search_path, image_name + ext)
                if os.path.exists(image_path):
                    return image_path
    
    return None

def predict_single_image(image_path, model, runs_dir, seg_dir):
    """Predict on a single image."""
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = test_transform(image).unsqueeze(0).to(DEVICE)
    
    # Get image ID from filename
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    
    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor, task_type='both')
        
        # Extract classification and segmentation outputs
        class_logits = outputs['classification']
        seg_logits = outputs['segmentation']
        
        class_pred = torch.argmax(class_logits, dim=1).cpu().numpy()[0]
        seg_preds = torch.sigmoid(seg_logits).cpu().numpy()[0]  # Shape: (1, 512, 512)
        
        # If segmentation output has only 1 channel, we need to create 5 channels for 5 lesion types
        if seg_preds.shape[0] == 1:
            # Duplicate the single channel to create 5 channels for different lesion types
            seg_preds = np.repeat(seg_preds, 5, axis=0)
    
    # Save individual segmentation masks
    lesion_types = ['Microaneurysms', 'Haemorrhages', 'Hard_Exudates', 'Soft_Exudates', 'Optic_Disc']
    saved_masks = []
    
    for j, lesion_type in enumerate(lesion_types):
        mask = seg_preds[j]
        mask_path = save_segmentation_mask(mask, image_id, lesion_type, seg_dir)
        saved_masks.append(mask_path)
    
    # Save combined segmentation visualization
    combined_path = save_combined_segmentation(seg_preds, image_id, seg_dir)
    
    # Save original image for reference
    original_save_path = os.path.join(seg_dir, f"{image_id}_original.png")
    image.save(original_save_path)
    
    result = {
        'image_id': image_id,
        'image_path': image_path,
        'retinopathy_grade_pred': int(class_pred),
        'segmentation_masks': saved_masks,
        'combined_visualization': combined_path,
        'original_image': original_save_path
    }
    
    return result

def main():
    """
    Main function for running inference on a single image.
    """
    parser = argparse.ArgumentParser(description='Run inference on a single image from IDRiD dataset')
    parser.add_argument('--image', type=str, required=True, help='Path to single image or image name to find in dataset')
    args = parser.parse_args()
    
    # Step 1: Create runs folder structure
    runs_dir, seg_dir = create_runs_folder()
    print(f"Created runs directory: {runs_dir}")
    
    # Step 2: Load trained model
    print("Loading model...")
    model = ModularMultiTaskModel(num_classes_classification=5, num_experts=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")
    
    # Step 3: Find the image
    print(f"Looking for image: {args.image}")
    
    # Check if it's a full path or just image name
    if os.path.exists(args.image):
        image_path = args.image
        print(f"Found image at: {image_path}")
    else:
        # Try to find the image in the dataset
        image_path = find_image_in_dataset(args.image)
        if not image_path:
            print(f"Error: Could not find image '{args.image}' in dataset or as full path")
            print("Please provide a valid image path or image name from the dataset.")
            return
        print(f"Found image in dataset at: {image_path}")
    
    # Step 4: Run prediction on single image
    print("Running inference...")
    result = predict_single_image(image_path, model, runs_dir, seg_dir)
    
    # Step 5: Print results
    print(f"\n{'='*50}")
    print(f"PREDICTION RESULTS")
    print(f"{'='*50}")
    print(f"Image: {result['image_id']}")
    print(f"Predicted Retinopathy Grade: {result['retinopathy_grade_pred']}")
    print(f"Original image saved to: {result['original_image']}")
    print(f"Combined segmentation saved to: {result['combined_visualization']}")
    print(f"Individual masks saved to: {seg_dir}")
    
    # Save single result to JSON
    import json
    summary_file = os.path.join(runs_dir, "single_prediction_result.json")
    with open(summary_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to: {runs_dir}")
    print(f"Summary file: {summary_file}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 