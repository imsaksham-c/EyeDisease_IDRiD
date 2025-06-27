import torch
import random
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class Config:
    # Data settings
    data_root: str = "dataset/"
    image_size: Tuple[int, int] = (512, 512)
    batch_size: int = 16
    num_workers: int = 4

    # Model settings
    backbone: str = "resnet50"
    num_classes: int = 5  # Disease grading classes
    num_experts: int = 3
    hidden_dim: int = 256
    dropout_rate: float = 0.2

    # Training settings
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 10

    # Loss weights
    classification_weight: float = 1.0
    segmentation_weight: float = 1.0

    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducibility
    seed: int = 42

    # Paths
    checkpoint_dir: str = "checkpoints/"
    results_dir: str = "results/"


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
