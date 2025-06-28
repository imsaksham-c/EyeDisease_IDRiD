"""
predict.py
----------
Script for running inference on the IDRiD dataset using a trained model. Loads the model, prepares the test dataset, and outputs predictions for classification and segmentation tasks.
"""

import torch
from utils.models import ModularMultiTaskModel
from utils.data_utils import EyeDiseaseDataset
from utils.dataset_reorganiser import DatasetReorganizer
from torchvision import transforms
from torch.utils.data import DataLoader
import os

# Configuration
DATASET_PATH = "dataset"
MODEL_PATH = "best_model.pth"
BATCH_SIZE = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define test transforms (should match validation transforms used in training)
test_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

def main():
    """
    Main function for running inference on the test set.
    """
    # Step 1: Prepare test data
    reorganizer = DatasetReorganizer(DATASET_PATH)
    reorganized_data = reorganizer.reorganize_data()
    test_dataset = EyeDiseaseDataset(
        reorganized_data['test'],
        transform=test_transform,
        task_type='both'
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Step 2: Load trained model
    model = ModularMultiTaskModel(num_classes_classification=5, num_experts=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Step 3: Run inference
    results = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(DEVICE)
            image_ids = batch['image_id']
            outputs = model(images)
            # Assuming model returns a tuple: (classification_logits, segmentation_logits)
            class_logits, seg_logits = outputs
            class_preds = torch.argmax(class_logits, dim=1).cpu().numpy()
            seg_preds = torch.sigmoid(seg_logits).cpu().numpy()
            for i, img_id in enumerate(image_ids):
                results.append({
                    'image_id': img_id,
                    'retinopathy_grade_pred': int(class_preds[i]),
                    'segmentation_pred': seg_preds[i],
                })

    # Step 4: Output predictions (example: print or save to file)
    for res in results:
        print(f"Image: {res['image_id']} | Predicted Grade: {res['retinopathy_grade_pred']}")
        # Optionally, save segmentation masks or further process results

if __name__ == "__main__":
    main() 