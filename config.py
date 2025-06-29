# Configuration for Multi-Task Learning System
import os
import torch

class Config:
    """Configuration class for the multi-task learning system"""
    
    def __init__(self):
        # Paths
        self.dataset_root = "dataset"
        self.seg_data_path = os.path.join(self.dataset_root, "A. Segmentation")
        self.cls_data_path = os.path.join(self.dataset_root, "B. Disease Grading")
        self.checkpoint_dir = "checkpoints"
        self.runs_dir = "runs"
        self.log_dir = "logs"
        
        # Model parameters
        self.backbone = "resnet50"
        self.pretrained = True
        self.num_classes_dr = 5  # DR grades 0-4
        self.num_classes_dme = 3  # DME risk 0-2
        self.num_seg_classes = 6  # 5 lesion types + background
        self.dropout_rate = 0.3
        
        # Training parameters
        self.batch_size = 8
        self.num_epochs = 100
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.early_stopping_patience = 15
        self.save_best_only = True
        
        # Image parameters
        self.image_size = 512
        self.input_channels = 3
        
        # Loss weights
        self.cls_loss_weight = 1.0
        self.seg_loss_weight = 1.0
        self.gating_loss_weight = 0.1
        self.dice_loss_weight = 0.5
        self.bce_loss_weight = 0.5
        
        # Data augmentation
        self.use_augmentation = True
        self.rotation_limit = 15
        self.brightness_limit = 0.1
        self.contrast_limit = 0.1
        self.hue_shift_limit = 0.1
        self.sat_shift_limit = 0.1
        
        # Dataset balancing
        self.use_balanced_sampling = True  # Enable weighted random sampling
        self.use_class_weights = True      # Use class weights in loss functions
        self.lesion_boost_factor = 0.2     # Boost factor for samples with lesions
        self.task_balance_factor = 1.0     # Balance factor between seg and cls tasks
        
        # Advanced augmentation for imbalanced classes
        self.use_advanced_augmentation = True
        self.mixup_alpha = 0.2             # Mixup augmentation strength
        self.cutmix_prob = 0.3             # CutMix probability
        self.elastic_transform_prob = 0.2  # Elastic transform probability
        
        # Miscellaneous
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 4
        self.pin_memory = True
        self.random_seed = 42
        self.mixed_precision = True
        
        # Logging
        self.use_wandb = False  # Set to True if you want to use wandb
        self.log_every_n_steps = 10
        self.save_predictions = True
        self.save_visualizations = True
        
        # Gating parameters
        self.gating_hidden_dim = 256
        self.gating_temperature = 1.0
        
    def create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.runs_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
    def __str__(self):
        """String representation of config"""
        return f"Config(batch_size={self.batch_size}, lr={self.learning_rate}, epochs={self.num_epochs}, balanced={self.use_balanced_sampling})"