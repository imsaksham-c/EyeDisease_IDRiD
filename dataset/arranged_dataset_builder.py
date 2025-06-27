import os
import shutil
import random
import pandas as pd

# Set random seed for reproducibility
random.seed(42)

# Source directories
SEG_IMG_TRAIN = 'dataset/A. Segmentation/1. Original Images/a. Training Set'
SEG_IMG_TEST = 'dataset/A. Segmentation/1. Original Images/b. Testing Set'
SEG_MASK_TRAIN = 'dataset/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set'
SEG_MASK_TEST = 'dataset/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set'
GRAD_IMG_TRAIN = 'dataset/B. Disease Grading/1. Original Images/a. Training Set'
GRAD_IMG_TEST = 'dataset/B. Disease Grading/1. Original Images/b. Testing Set'
GRAD_LABEL_TRAIN = 'dataset/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv'
GRAD_LABEL_TEST = 'dataset/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv'

# Target directories
TARGET_ROOT = 'dataset/arranged'
SEG_IMG_DIR = os.path.join(TARGET_ROOT, 'segmentation', 'images')
SEG_MASK_DIR = os.path.join(TARGET_ROOT, 'segmentation', 'masks')
CLS_IMG_DIR = os.path.join(TARGET_ROOT, 'classification', 'images')
CLS_LABEL_DIR = os.path.join(TARGET_ROOT, 'classification', 'labels')

SPLITS = ['train', 'val', 'test']
SEGMENTATION_TYPES = [
    ("Microaneurysms", "MA"),
    ("Haemorrhages", "HE"),
    ("Hard Exudates", "EX"),
    ("Soft Exudates", "SE"),
    ("Optic Disc", "OD"),
]

# 1. Create directory structure
def make_dirs():
    for split in SPLITS:
        os.makedirs(os.path.join(SEG_IMG_DIR, split), exist_ok=True)
        os.makedirs(os.path.join(SEG_MASK_DIR, split), exist_ok=True)
        os.makedirs(os.path.join(CLS_IMG_DIR, split), exist_ok=True)
    os.makedirs(CLS_LABEL_DIR, exist_ok=True)

# 2. Get segmentation and classification image names
def get_segmentation_image_names():
    train = sorted([os.path.splitext(f)[0] for f in os.listdir(SEG_IMG_TRAIN) if f.endswith('.jpg')])
    test = sorted([os.path.splitext(f)[0] for f in os.listdir(SEG_IMG_TEST) if f.endswith('.jpg')])
    return train, test

def get_classification_image_names():
    train = sorted([os.path.splitext(f)[0] for f in os.listdir(GRAD_IMG_TRAIN) if f.endswith('.jpg')])
    test = sorted([os.path.splitext(f)[0] for f in os.listdir(GRAD_IMG_TEST) if f.endswith('.jpg')])
    return train, test

# 3. Split train into train/val (80/20)
def split_train_val(names, val_ratio=0.2):
    n = len(names)
    n_val = int(n * val_ratio)
    shuffled = names[:]
    random.shuffle(shuffled)
    val_names = sorted(shuffled[:n_val])
    train_names = sorted(shuffled[n_val:])
    return train_names, val_names

# 4. Copy segmentation images and masks
def copy_segmentation_images(image_names, src_dir, split):
    for name in image_names:
        src_path = os.path.join(src_dir, name + '.jpg')
        dst_path = os.path.join(SEG_IMG_DIR, split, name + '.jpg')
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"[Warning] Segmentation image not found: {src_path}")

def copy_segmentation_masks(image_names, src_mask_dir, split):
    for name in image_names:
        for lesion, suffix in SEGMENTATION_TYPES:
            mask_path = os.path.join(src_mask_dir, f"{SEGMENTATION_TYPES.index((lesion, suffix))+1}. {lesion}", f"{name}_{suffix}.tif")
            dst_path = os.path.join(SEG_MASK_DIR, split, f"{name}_{suffix}.tif")
            if os.path.exists(mask_path):
                shutil.copy(mask_path, dst_path)

# 5. Copy classification images and create label CSVs
def copy_classification_images(image_names, src_dir, split):
    for name in image_names:
        src_path = os.path.join(src_dir, name + '.jpg')
        dst_path = os.path.join(CLS_IMG_DIR, split, name + '.jpg')
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"[Warning] Classification image not found: {src_path}")

def create_classification_label_csv(image_names, label_csv, split):
    if not os.path.exists(label_csv):
        print(f"[Warning] Label CSV not found for {split}")
        return
    df = pd.read_csv(label_csv)
    # Find the correct columns for image name and grade
    name_col = [c for c in df.columns if 'name' in c.lower()][0]
    grade_col = [c for c in df.columns if 'retinopathy' in c.lower()][0]
    df[name_col] = df[name_col].astype(str).str.strip()
    df = df[df[name_col].isin(image_names)]
    df = df[[name_col, grade_col]]
    df.columns = ['image_name', 'grade']
    df.to_csv(os.path.join(CLS_LABEL_DIR, f"{split}.csv"), index=False)

if __name__ == "__main__":
    make_dirs()
    # Segmentation
    seg_train, seg_test = get_segmentation_image_names()
    seg_train, seg_val = split_train_val(seg_train, val_ratio=0.2)
    copy_segmentation_images(seg_train, SEG_IMG_TRAIN, 'train')
    copy_segmentation_images(seg_val, SEG_IMG_TRAIN, 'val')
    copy_segmentation_images(seg_test, SEG_IMG_TEST, 'test')
    copy_segmentation_masks(seg_train, SEG_MASK_TRAIN, 'train')
    copy_segmentation_masks(seg_val, SEG_MASK_TRAIN, 'val')
    copy_segmentation_masks(seg_test, SEG_MASK_TEST, 'test')
    # Classification
    cls_train, cls_test = get_classification_image_names()
    cls_train, cls_val = split_train_val(cls_train, val_ratio=0.2)
    copy_classification_images(cls_train, GRAD_IMG_TRAIN, 'train')
    copy_classification_images(cls_val, GRAD_IMG_TRAIN, 'val')
    copy_classification_images(cls_test, GRAD_IMG_TEST, 'test')
    create_classification_label_csv(cls_train, GRAD_LABEL_TRAIN, 'train')
    create_classification_label_csv(cls_val, GRAD_LABEL_TRAIN, 'val')
    create_classification_label_csv(cls_test, GRAD_LABEL_TEST, 'test')
    print("Segmentation and classification datasets arranged separately at ./dataset/arranged/") 