# Multi-task model architecture with gating mechanism
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import logging

logger = logging.getLogger(__name__)

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # Spatial attention
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        
    def forward(self, x):
        # Channel attention
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.conv(spatial_att))
        x = x * spatial_att
        
        return x

class SharedBackbone(nn.Module):
    """Shared ResNet-50 backbone"""
    def __init__(self, pretrained=True):
        super(SharedBackbone, self).__init__()
        
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
            self.resnet = models.resnet50(weights=weights)
        else:
            self.resnet = models.resnet50(weights=None)
        
        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # Get feature dimensions from each layer
        self.layer_dims = {
            'layer1': 256,
            'layer2': 512, 
            'layer3': 1024,
            'layer4': 2048
        }
        
    def forward(self, x):
        """Extract multi-scale features"""
        features = {}
        
        # Initial layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        # Extract features from each layer
        x = self.resnet.layer1(x)
        features['layer1'] = x
        
        x = self.resnet.layer2(x)
        features['layer2'] = x
        
        x = self.resnet.layer3(x)
        features['layer3'] = x
        
        x = self.resnet.layer4(x)
        features['layer4'] = x
        
        return features

class GatingNetwork(nn.Module):
    """Dynamic gating network for task routing"""
    def __init__(self, feature_dim=2048, hidden_dim=256, num_tasks=2):
        super(GatingNetwork, self).__init__()
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_gate = nn.Linear(hidden_dim, num_tasks)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, features):
        """Compute gating weights"""
        # Global average pooling
        x = self.global_pool(features['layer4'])
        x = x.view(x.size(0), -1)
        
        # MLP layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Gating weights (softmax)
        gates = F.softmax(self.fc_gate(x), dim=1)
        
        return gates

class ClassificationHead(nn.Module):
    """Classification head for DR grading and DME risk"""
    def __init__(self, feature_dim=2048, num_classes_dr=5, num_classes_dme=3, dropout=0.3):
        super(ClassificationHead, self).__init__()
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        
        # Shared feature compression
        self.fc_shared = nn.Linear(feature_dim, 512)
        
        # Task-specific heads
        self.fc_dr = nn.Linear(512, num_classes_dr)
        self.fc_dme = nn.Linear(512, num_classes_dme)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, features):
        """Forward pass for classification"""
        x = self.global_pool(features['layer4'])
        x = x.view(x.size(0), -1)
        
        # Shared representation
        x = self.relu(self.fc_shared(x))
        x = self.dropout(x)
        
        # Task-specific predictions
        dr_logits = self.fc_dr(x)
        dme_logits = self.fc_dme(x)
        
        return {
            'dr_logits': dr_logits,
            'dme_logits': dme_logits
        }

class SegmentationHead(nn.Module):
    """U-Net style segmentation head with CBAM attention"""
    def __init__(self, backbone_dims, num_classes=6):
        super(SegmentationHead, self).__init__()
        
        self.num_classes = num_classes
        
        # Decoder layers with skip connections
        self.upconv4 = nn.ConvTranspose2d(backbone_dims['layer4'], 512, 2, stride=2)
        self.conv4 = self._make_conv_block(512 + backbone_dims['layer3'], 512)
        self.cbam4 = CBAM(512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = self._make_conv_block(256 + backbone_dims['layer2'], 256)
        self.cbam3 = CBAM(256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = self._make_conv_block(128 + backbone_dims['layer1'], 128)
        self.cbam2 = CBAM(128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = self._make_conv_block(64, 64)
        
        # Final classifier
        self.final_conv = nn.Conv2d(64, num_classes, 1)
        
    def _make_conv_block(self, in_channels, out_channels):
        """Create a convolutional block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features):
        """Forward pass for segmentation"""
        # Start from the deepest features
        x = features['layer4']  # [B, 2048, H/32, W/32]
        
        # Decoder path with skip connections
        x = self.upconv4(x)  # [B, 512, H/16, W/16]
        x = torch.cat([x, features['layer3']], dim=1)
        x = self.conv4(x)
        x = self.cbam4(x)
        
        x = self.upconv3(x)  # [B, 256, H/8, W/8]
        x = torch.cat([x, features['layer2']], dim=1)
        x = self.conv3(x)
        x = self.cbam3(x)
        
        x = self.upconv2(x)  # [B, 128, H/4, W/4]
        x = torch.cat([x, features['layer1']], dim=1)
        x = self.conv2(x)
        x = self.cbam2(x)
        
        x = self.upconv1(x)  # [B, 64, H/2, W/2]
        x = self.conv1(x)
        
        # Upsample to original size
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Final classification
        logits = self.final_conv(x)  # [B, num_classes, H, W]
        
        return logits

class MultiTaskModel(nn.Module):
    """Complete multi-task model with gating"""
    def __init__(self, config):
        super(MultiTaskModel, self).__init__()
        
        self.config = config
        
        # Shared backbone
        self.backbone = SharedBackbone(pretrained=config.pretrained)
        
        # Gating network
        self.gating = GatingNetwork(
            feature_dim=self.backbone.layer_dims['layer4'],
            hidden_dim=config.gating_hidden_dim,
            num_tasks=2  # classification and segmentation
        )
        
        # Task-specific heads
        self.classification_head = ClassificationHead(
            feature_dim=self.backbone.layer_dims['layer4'],
            num_classes_dr=config.num_classes_dr,
            num_classes_dme=config.num_classes_dme,
            dropout=config.dropout_rate
        )
        
        self.segmentation_head = SegmentationHead(
            backbone_dims=self.backbone.layer_dims,
            num_classes=config.num_seg_classes
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, task_type=None):
        """Forward pass with task routing"""
        batch_size = x.size(0)
        
        # Extract features from backbone
        features = self.backbone(x)
        
        # Compute gating weights
        gates = self.gating(features)  # [B, 2] - [cls_gate, seg_gate]
        
        # Task-specific predictions
        cls_output = self.classification_head(features)
        seg_output = self.segmentation_head(features)
        
        # Ensure segmentation output matches target size
        if seg_output.size(-1) != self.config.image_size:
            seg_output = F.interpolate(
                seg_output, 
                size=(self.config.image_size, self.config.image_size),
                mode='bilinear', 
                align_corners=False
            )
        
        return {
            'cls_output': cls_output,
            'seg_output': seg_output,
            'gates': gates,
            'features': features
        }
    
    def get_num_parameters(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)