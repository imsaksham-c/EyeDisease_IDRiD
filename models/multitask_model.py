import torch
import torch.nn as nn
from .backbone import SharedBackbone
from .routing import DynamicRouter
from .experts import ClassificationExpert, SegmentationExpert, GeneralExpert

class MultiTaskModel(nn.Module):
    """Complete multi-task learning model."""
    
    def __init__(self, config):
        super().__init__()
        
        # Shared backbone
        self.backbone = SharedBackbone(config.backbone, pretrained=True)
        
        # Dynamic router
        self.router = DynamicRouter(
            self.backbone.feature_dim, 
            config.num_experts, 
            config.hidden_dim
        )
        
        # Expert modules
        self.experts = nn.ModuleList([
            ClassificationExpert(
                self.backbone.feature_dim, 
                config.num_classes, 
                config.hidden_dim, 
                config.dropout_rate
            ),
            SegmentationExpert(
                self.backbone.feature_dim, 
                config.dropout_rate
            ),
            GeneralExpert(
                self.backbone.feature_dim, 
                config.hidden_dim, 
                config.dropout_rate
            )
        ])
        
        # Task-specific heads
        self.classification_head = nn.Linear(config.hidden_dim, config.num_classes)
        
    def forward(self, x, task='both'):
        """
        Args:
            x: Input images
            task: 'classification', 'segmentation', or 'both'
        """
        # Extract shared features
        shared_features = self.backbone(x)
        
        outputs = {}
        
        if task in ['classification', 'both']:
            # Classification task
            routing_weights = self.router(shared_features, task_id=0)
            
            # Expert mixing for classification
            cls_output = None
            for i, expert in enumerate(self.experts):
                if i == 0:  # Classification expert
                    expert_output = expert(shared_features)
                    if cls_output is None:
                        cls_output = routing_weights[:, i:i+1] * expert_output
                    else:
                        cls_output += routing_weights[:, i:i+1] * expert_output
            
            outputs['classification'] = cls_output
            outputs['cls_routing'] = routing_weights
        
        if task in ['segmentation', 'both']:
            # Segmentation task
            routing_weights = self.router(shared_features, task_id=1)
            
            # Expert mixing for segmentation
            seg_output = None
            for i, expert in enumerate(self.experts):
                if i == 1:  # Segmentation expert
                    expert_output = expert(shared_features)
                    if seg_output is None:
                        seg_output = routing_weights[:, i:i+1].unsqueeze(-1).unsqueeze(-1) * expert_output
                    else:
                        seg_output += routing_weights[:, i:i+1].unsqueeze(-1).unsqueeze(-1) * expert_output
            
            outputs['segmentation'] = seg_output
            outputs['seg_routing'] = routing_weights
        
        return outputs