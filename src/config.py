import torch
import os

# --- Basic Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# --- Path Definitions ---
DATA_PATH = "preprocessed_data/"
OUTPUT_PATH = "outputs/"
MODEL_PATH = os.path.join(OUTPUT_PATH, "models/")
RESULTS_PATH = os.path.join(OUTPUT_PATH, "results/")

TRAIN_CSV = os.path.join(DATA_PATH, "train.csv")
TEST_CSV = os.path.join(DATA_PATH, "test.csv")

# --- Model Parameters ---
ENCODER = 'resnet34'
PRETRAINED = True
NUM_CLASSES_CLASSIFICATION = 5  # DR grades 0-4
NUM_CLASSES_SEGMENTATION = 5  # 4 lesion types + 1 Optic Disc

# --- Training Hyperparameters ---
IMAGE_SIZE = 512
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 10

# --- Loss Function Configuration ---
# Weighting for the combined loss: loss = (alpha * cls_loss) + ((1 - alpha) * seg_loss)
LOSS_ALPHA = 0.4 # Balances the two task losses.
