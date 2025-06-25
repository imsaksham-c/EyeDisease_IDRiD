import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicRouter(nn.Module):
    """Dynamic routing mechanism for expert selection."""
    
    def __init__(self, feature_dim, num_experts, hidden_dim=256):
        super().__init__()
        self.num_experts = num_experts
        
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=1)
        )
    
    def forward(self, features, task_id=None):
        """
        Args:
            features: Shared features from backbone
            task_id: Optional task identifier (0: classification, 1: segmentation)
        Returns:
            routing_weights: Weights for expert selection
        """
        routing_weights = self.router(features)
        
        # Task-specific routing bias (optional)
        if task_id is not None:
            task_bias = torch.zeros_like(routing_weights)
            if task_id == 0:  # Classification
                task_bias[:, 0] += 0.1
            elif task_id == 1:  # Segmentation
                task_bias[:, 1] += 0.1
            routing_weights = F.softmax(routing_weights + task_bias, dim=1)
        
        return routing_weights 