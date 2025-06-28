import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
from collections import defaultdict
import random

# --- Configuration ---
CONFIG = {
    "base_data_path": "data/IDRiD",
    "output_path": "preprocessed_data",
    "image_size": (512, 512),
    "train_test_split": 0.8,  # 80% for training, 20% for testing
    "random_seed": 42,  # For reproducible splits
    "lesion_map": {
        "MA": 1, "HE": 2, "EX": 3, "SE": 4, "OD": 5
    },
    "mask_folders": {
        "MA": "1. Microaneurysms", "HE": "2. Haemorrhages",
        "EX": "3. Hard Exudates", "SE": "4. Soft Exudates", "OD": "5. Optic Disc"
    }
}

# --- Helper Functions ---
def process_and_save_image(img_path, output_folder, new_id):
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"OpenCV could not read image at path {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, CONFIG["image_size"], interpolation=cv2.INTER_AREA)
        save_path = os.path.join(output_folder, f"{new_id}.png")
        cv2.imwrite(save_path, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
        return os.path.join(os.path.basename(output_folder), f"{new_id}.png")
    except Exception as e:
        print(f"Error during image processing for {img_path}: {e}")
        return None

def process_and_save_masks(mask_paths, output_folder, new_id):
    num_channels = len(CONFIG["lesion_map"])
    combined_mask = np.zeros((*CONFIG["image_size"], num_channels), dtype=np.uint8)
    
    processed_lesions = []
    for lesion_abbr, class_idx in CONFIG["lesion_map"].items():
        mask_path = next((p for p in mask_paths if f"_{lesion_abbr}.tif" in os.path.basename(p)), None)
        if mask_path:
            try:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    print(f"Warning: Could not read mask {mask_path}")
                    continue
                    
                mask_resized = cv2.resize(mask, CONFIG["image_size"], interpolation=cv2.INTER_NEAREST)
                _, mask_binary = cv2.threshold(mask_resized, 1, 255, cv2.THRESH_BINARY)
                combined_mask[:, :, class_idx - 1] = mask_binary / 255
                processed_lesions.append(lesion_abbr)
            except Exception as e:
                print(f"Error processing mask {mask_path}: {e}")
    
    if processed_lesions:
        save_path = os.path.join(output_folder, f"{new_id}_mask.npy")
        np.save(save_path, combined_mask)
        return os.path.join(os.path.basename(output_folder), f"{new_id}_mask.npy")
    else:
        print(f"Warning: No valid masks found for {new_id}")
        return None

def get_image_id(filepath):
    """Extracts a standardized ID (e.g., IDRiD_001) from a filepath."""
    match = re.search(r'IDRiD_(\d+)', os.path.basename(filepath))
    if match:
        return f"IDRiD_{int(match.group(1)):03d}"
    return None

def validate_processed_data(train_df, test_df, output_path):
    """Validates the processed data for common issues."""
    print("\n=== Data Validation ===")
    
    # Check for missing files
    missing_train_images = 0
    missing_test_images = 0
    missing_train_masks = 0
    missing_test_masks = 0
    
    for _, row in train_df.iterrows():
        img_path = os.path.join(output_path, row['image_path'])
        if not os.path.exists(img_path):
            missing_train_images += 1
        
        if pd.notna(row['mask_path']):
            mask_path = os.path.join(output_path, row['mask_path'])
            if not os.path.exists(mask_path):
                missing_train_masks += 1
    
    for _, row in test_df.iterrows():
        img_path = os.path.join(output_path, row['image_path'])
        if not os.path.exists(img_path):
            missing_test_images += 1
        
        if pd.notna(row['mask_path']):
            mask_path = os.path.join(output_path, row['mask_path'])
            if not os.path.exists(mask_path):
                missing_test_masks += 1
    
    print(f"Missing train images: {missing_train_images}")
    print(f"Missing test images: {missing_test_images}")
    print(f"Missing train masks: {missing_train_masks}")
    print(f"Missing test masks: {missing_test_masks}")
    
    # Check class distribution
    print(f"\nTrain DR Grade distribution: {dict(train_df['dr_grade'].value_counts().sort_index())}")
    print(f"Test DR Grade distribution: {dict(test_df['dr_grade'].value_counts().sort_index())}")
    
    # Check mask availability
    train_with_masks = train_df['mask_path'].notna().sum()
    test_with_masks = test_df['mask_path'].notna().sum()
    print(f"\nTrain images with masks: {train_with_masks}/{len(train_df)} ({train_with_masks/len(train_df)*100:.1f}%)")
    print(f"Test images with masks: {test_with_masks}/{len(test_df)} ({test_with_masks/len(test_df)*100:.1f}%)")
    
    return missing_train_images == 0 and missing_test_images == 0 and missing_train_masks == 0 and missing_test_masks == 0

# --- Main Preprocessing Logic ---
def main():
    print("Starting dataset preprocessing...")
    
    # Set random seed for reproducible splits
    random.seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])
    
    base_path = CONFIG["base_data_path"]
    output_path = CONFIG["output_path"]

    for sub_dir in ["train_images", "test_images", "train_masks", "test_masks"]:
        os.makedirs(os.path.join(output_path, sub_dir), exist_ok=True)

    # 1. Load Classification Labels
    cls_grading_path = os.path.join(base_path, "B. Disease Grading")
    try:
        train_labels_df = pd.read_csv(os.path.join(cls_grading_path, "2. Groundtruths", "a. IDRiD_Disease Grading_Training Labels.csv"))
        test_labels_df = pd.read_csv(os.path.join(cls_grading_path, "2. Groundtruths", "b. IDRiD_Disease Grading_Testing Labels.csv"))
    except FileNotFoundError as e:
        print(f"FATAL: Label CSV file not found: {e.filename}. Please check your dataset structure.")
        return

    # Standardize column names and create a lookup dictionary
    def get_cls_lookup(df):
        dme_col = next((col for col in df.columns if "macular edema" in col.lower()), None)
        df.rename(columns={"Image name": "id", "Retinopathy grade": "dr_grade", dme_col: "dme_risk"}, inplace=True)
        return df.set_index('id').to_dict('index')
    
    train_cls_lookup = get_cls_lookup(train_labels_df)
    test_cls_lookup = get_cls_lookup(test_labels_df)

    # 2. Gather all files and associate them with their IDs
    all_files = defaultdict(lambda: {'split': None, 'img_path': None, 'mask_paths': []})
    
    # FIX: Prioritize segmentation dataset for images that have masks
    # Process segmentation dataset first (has both images and masks)
    for split, prefix in [('train', 'a.'), ('test', 'b.')]:
        set_name = f"{prefix} {split.capitalize()}ing Set"
        
        # Segmentation images and masks
        seg_img_dir = os.path.join(base_path, "A. Segmentation", "1. Original Images", set_name)
        seg_mask_dir = os.path.join(base_path, "A. Segmentation", "2. All Segmentation Groundtruths", set_name)
        
        if os.path.isdir(seg_img_dir):
            for f in os.listdir(seg_img_dir):
                img_id = get_image_id(f)
                if img_id:
                    all_files[img_id]['split'] = split
                    all_files[img_id]['img_path'] = os.path.join(seg_img_dir, f)
        
        if os.path.isdir(seg_mask_dir):
            for lesion_folder in os.listdir(seg_mask_dir):
                lesion_path = os.path.join(seg_mask_dir, lesion_folder)
                if os.path.isdir(lesion_path):
                    for f in os.listdir(lesion_path):
                        img_id = get_image_id(f)
                        if img_id:
                            all_files[img_id]['mask_paths'].append(os.path.join(lesion_path, f))
        
        # Only add classification images if they don't already have segmentation data
        cls_img_dir = os.path.join(cls_grading_path, "1. Original Images", set_name)
        if os.path.isdir(cls_img_dir):
            for f in os.listdir(cls_img_dir):
                img_id = get_image_id(f)
                if img_id and not all_files[img_id]['img_path']: # Only add if not already present
                    all_files[img_id]['split'] = split
                    all_files[img_id]['img_path'] = os.path.join(cls_img_dir, f)

    # 3. Process the master file list
    master_data_list = []
    print(f"Found {len(all_files)} unique images across all datasets. Processing...")
    
    # FIX: Split data more intelligently - use images with masks for training when possible
    images_with_masks = []
    images_without_masks = []
    
    for img_id, data in all_files.items():
        if not data['img_path'] or not data['split']:
            print(f"Warning: Skipping ID {img_id} due to missing image file or split information.")
            continue
        
        if data['mask_paths']:
            images_with_masks.append((img_id, data))
        else:
            images_without_masks.append((img_id, data))
    
    print(f"Images with masks: {len(images_with_masks)}")
    print(f"Images without masks: {len(images_without_masks)}")
    
    # Sort images with masks for reproducible splits
    images_with_masks.sort(key=lambda x: x[0])  # Sort by image ID
    
    # Use configurable split ratio for images with masks
    train_mask_count = int(len(images_with_masks) * CONFIG["train_test_split"])
    train_with_masks = images_with_masks[:train_mask_count]
    test_with_masks = images_with_masks[train_mask_count:]
    
    # Use remaining images without masks for training (classification only)
    train_without_masks = images_without_masks
    
    # Combine all training data
    all_train_data = train_with_masks + train_without_masks
    all_test_data = test_with_masks
    
    print(f"Training set: {len(all_train_data)} images ({len(train_with_masks)} with masks, {len(train_without_masks)} without masks)")
    print(f"Test set: {len(all_test_data)} images ({len(test_with_masks)} with masks)")
    
    # Process training data
    for img_id, data in tqdm(all_train_data, desc="Processing training data"):
        out_img_dir = os.path.join(output_path, "train_images")
        out_mask_dir = os.path.join(output_path, "train_masks")
        
        # Process image
        img_rel_path = process_and_save_image(data['img_path'], out_img_dir, img_id)
        if not img_rel_path:
            continue
            
        # Process masks
        mask_rel_path = None
        if data['mask_paths']:
            mask_rel_path = process_and_save_masks(data['mask_paths'], out_mask_dir, img_id)
            
        # Get classification labels
        cls_lookup = train_cls_lookup
        cls_data = cls_lookup.get(img_id.strip())
        
        dr_grade = cls_data['dr_grade'] if cls_data else -1
        dme_risk = cls_data['dme_risk'] if cls_data else -1
            
        master_data_list.append({
            "id": img_id,
            "split": "train",
            "image_path": img_rel_path,
            "mask_path": mask_rel_path,
            "dr_grade": dr_grade,
            "dme_risk": dme_risk,
        })
    
    # Process test data
    for img_id, data in tqdm(all_test_data, desc="Processing test data"):
        out_img_dir = os.path.join(output_path, "test_images")
        out_mask_dir = os.path.join(output_path, "test_masks")
        
        # Process image
        img_rel_path = process_and_save_image(data['img_path'], out_img_dir, img_id)
        if not img_rel_path:
            continue
            
        # Process masks
        mask_rel_path = None
        if data['mask_paths']:
            mask_rel_path = process_and_save_masks(data['mask_paths'], out_mask_dir, img_id)
            
        # Get classification labels
        cls_lookup = test_cls_lookup
        cls_data = cls_lookup.get(img_id.strip())
        
        dr_grade = cls_data['dr_grade'] if cls_data else -1
        dme_risk = cls_data['dme_risk'] if cls_data else -1
            
        master_data_list.append({
            "id": img_id,
            "split": "test",
            "image_path": img_rel_path,
            "mask_path": mask_rel_path,
            "dr_grade": dr_grade,
            "dme_risk": dme_risk,
        })

    # 4. Create final dataframes and save
    if not master_data_list:
        print("FATAL: No data was processed. Please check your dataset structure and file paths.")
        return

    final_df = pd.DataFrame(master_data_list)
    train_df = final_df[final_df['split'] == 'train'].drop(columns=['split'])
    test_df = final_df[final_df['split'] == 'test'].drop(columns=['split'])

    train_df.to_csv(os.path.join(output_path, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_path, "test.csv"), index=False)

    print("\nPreprocessing finished successfully!")
    print(f"Total images processed: {len(final_df)}")
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    print(f"Unified CSVs saved to: {output_path}")

    # Validate processed data
    if validate_processed_data(train_df, test_df, output_path):
        print("\nData validation passed!")
    else:
        print("\nData validation failed. Please check your dataset structure and file paths.")

if __name__ == "__main__":
    main()
