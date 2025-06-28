"""
models.py
---------
Defines neural network architectures for multi-task and single-task learning on the IDRiD dataset, including modular and expert-based models for classification and segmentation tasks.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ExpertModule(nn.Module):
    """
    Expert module for Mixture-of-Experts architecture.
    Used for both classification and segmentation expert branches.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super(ExpertModule, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

class GatingNetwork(nn.Module):
    """
    Gating network to compute expert weights for dynamic routing.
    """
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_experts),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.gate(x)

class ModularMultiTaskModel(nn.Module):
    """
    Modular multi-task model with shared ResNet-50 backbone, expert modules, and gating networks.
    Supports both classification and segmentation tasks.
    """
    def __init__(self, num_classes_classification=5, num_experts=3, hidden_dim=512):
        super(ModularMultiTaskModel, self).__init__()
        # Shared backbone (ResNet-50, pretrained)
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone_features = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.feature_dim = self.backbone_features
        # Expert modules for classification
        self.classification_experts = nn.ModuleList([
            ExpertModule(self.feature_dim, hidden_dim, num_classes_classification)
            for _ in range(num_experts)
        ])
        # Expert modules for segmentation
        self.segmentation_experts = nn.ModuleList([
            self._create_segmentation_expert()
            for _ in range(num_experts)
        ])
        # Gating networks
        self.classification_gate = GatingNetwork(self.feature_dim, num_experts)
        self.segmentation_gate = GatingNetwork(self.feature_dim, num_experts)
        # Task selector (learns to route between tasks)
        self.task_selector = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 2),
            nn.Sigmoid()
        )
    def _create_segmentation_expert(self):
        """
        Creates a segmentation expert branch (decoder) using ConvTranspose2d layers.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(self.feature_dim, 512, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid()
        )
    def forward(self, x, task_type='both'):
        batch_size = x.size(0)
        # Extract features using shared backbone
        features = self.backbone(x)
        features_flat = features.view(batch_size, -1)
        # Task routing weights
        task_weights = self.task_selector(features_flat)
        outputs = {}
        if task_type in ['both', 'classification']:
            # Classification path
            cls_gate_weights = self.classification_gate(features_flat)
            cls_expert_outputs = []
            for expert in self.classification_experts:
                expert_out = expert(features_flat)
                cls_expert_outputs.append(expert_out)
            cls_expert_outputs = torch.stack(cls_expert_outputs, dim=2)
            cls_gate_weights = cls_gate_weights.unsqueeze(1)
            classification_output = torch.sum(cls_expert_outputs * cls_gate_weights, dim=2)
            outputs['classification'] = classification_output
            outputs['cls_gate_weights'] = cls_gate_weights.squeeze()
        if task_type in ['both', 'segmentation']:
            # Segmentation path
            seg_gate_weights = self.segmentation_gate(features_flat)
            seg_expert_outputs = []
            # Reshape features for segmentation experts
            features_2d = features.view(batch_size, self.feature_dim, 1, 1)
            for expert in self.segmentation_experts:
                expert_out = expert(features_2d)
                seg_expert_outputs.append(expert_out)
            seg_expert_outputs = torch.stack(seg_expert_outputs, dim=1)
            seg_gate_weights = seg_gate_weights.view(batch_size, -1, 1, 1, 1)
            segmentation_output = torch.sum(seg_expert_outputs * seg_gate_weights, dim=1)
            outputs['segmentation'] = segmentation_output
            outputs['seg_gate_weights'] = seg_gate_weights.squeeze()
        outputs['task_weights'] = task_weights
        return outputs

class SingleTaskModel(nn.Module):
    """
    Single-task model for either classification or segmentation.
    Uses ResNet-50 backbone and a task-specific head.
    """
    def __init__(self, num_classes=5, task_type='classification'):
        super(SingleTaskModel, self).__init__()
        self.task_type = task_type
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone_features = self.backbone.fc.in_features
        if task_type == 'classification':
            self.backbone.fc = nn.Linear(self.backbone_features, num_classes)
        elif task_type == 'segmentation':
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            self.segmentation_head = nn.Sequential(
                nn.ConvTranspose2d(self.backbone_features, 512, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, 4, 2, 1),
                nn.Sigmoid()
            )
    def forward(self, x):
        if self.task_type == 'classification':
            return {'classification': self.backbone(x)}
        elif self.task_type == 'segmentation':
            features = self.backbone(x)
            seg_output = self.segmentation_head(features)
            return {'segmentation': seg_output} 