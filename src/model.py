import torch
import torch.nn as nn
import torchvision.models as models
from src import config

class SegmentationDecoder(nn.Module):
    """
    U-Net style decoder for the segmentation task.
    It takes features from multiple levels of a ResNet encoder and upsamples them,
    using skip connections to preserve spatial details.
    """
    def __init__(self, encoder_channels, out_channels):
        super().__init__()
        
        # The channels from the ResNet-34 encoder are [64, 64, 128, 256, 512]
        # These correspond to the outputs of [base, layer1, layer2, layer3, layer4]
        
        # Bottom-up path (upsampling)
        self.upconv4 = nn.ConvTranspose2d(encoder_channels[4], 256, kernel_size=2, stride=2)
        self.decoder_block4 = self._decoder_block(256 + encoder_channels[3], 256) # Concat with layer3 output
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_block3 = self._decoder_block(128 + encoder_channels[2], 128) # Concat with layer2 output
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_block2 = self._decoder_block(64 + encoder_channels[1], 64) # Concat with layer1 output
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder_block1 = self._decoder_block(32 + encoder_channels[0], 32) # Concat with initial feature map
        
        # Final convolution to produce the output mask
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, encoder_features):
        # Unpack the features from the skip connections
        f0, f1, f2, f3, f4 = encoder_features
        
        # Decoder path
        x = self.upconv4(f4)
        x = torch.cat([x, f3], dim=1)
        x = self.decoder_block4(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, f2], dim=1)
        x = self.decoder_block3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, f1], dim=1)
        x = self.decoder_block2(x)

        x = self.upconv1(x)
        x = torch.cat([x, f0], dim=1)
        x = self.decoder_block1(x)
        
        # Final upsampling to match the original image size
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        return self.final_conv(x)


class MultiTaskModel(nn.Module):
    """
    The main multi-task model with a shared ResNet encoder and two separate heads
    for classification and segmentation. This version has a corrected forward pass.
    """
    def __init__(self, num_cls_classes, num_seg_classes, pretrained=True):
        super().__init__()
        
        # --- Shared Encoder ---
        weights = 'ResNet34_Weights.DEFAULT' if pretrained else None
        encoder = models.resnet34(weights=weights)
        
        # Explicitly define the encoder stages
        self.encoder_base = nn.Sequential(*list(encoder.children())[:3]) # First block: conv1, bn1, relu
        self.maxpool = encoder.maxpool
        self.encoder_layer1 = encoder.layer1
        self.encoder_layer2 = encoder.layer2
        self.encoder_layer3 = encoder.layer3
        self.encoder_layer4 = encoder.layer4
        
        encoder_channels = [64, 64, 128, 256, 512] # Channels after each stage

        # --- Classification Head ---
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(encoder_channels[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_cls_classes)
        )
        
        # --- Segmentation Head ---
        self.segmentation_head = SegmentationDecoder(
            encoder_channels=encoder_channels,
            out_channels=num_seg_classes
        )

    def forward(self, x):
        # --- Shared Encoder Forward Pass ---
        # This explicit flow ensures correct shapes for skip connections
        f0 = self.encoder_base(x)           # Shape: (B, 64, H/2, W/2) -> e.g., 256x256
        f1_in = self.maxpool(f0)
        f1 = self.encoder_layer1(f1_in)     # Shape: (B, 64, H/4, W/4) -> e.g., 128x128
        f2 = self.encoder_layer2(f1)        # Shape: (B, 128, H/8, W/8) -> e.g., 64x64
        f3 = self.encoder_layer3(f2)        # Shape: (B, 256, H/16, W/16) -> e.g., 32x32
        f4 = self.encoder_layer4(f3)        # Shape: (B, 512, H/32, W/32) -> e.g., 16x16
        
        # The encoder_features list now contains tensors with the correct spatial dimensions
        encoder_features = [f0, f1, f2, f3, f4]
        
        # --- Task-Specific Head Forward Passes ---
        # The classification head uses the deepest feature map
        classification_output = self.classification_head(f4)
        
        # The segmentation head uses all feature maps for its skip connections
        segmentation_output = self.segmentation_head(encoder_features)
        
        return {
            "classification": classification_output,
            "segmentation": segmentation_output
        }

def get_model():
    """Factory function to initialize and return the multi-task model."""
    model = MultiTaskModel(
        num_cls_classes=config.NUM_CLASSES_CLASSIFICATION,
        num_seg_classes=config.NUM_CLASSES_SEGMENTATION,
        pretrained=config.PRETRAINED
    )
    return model
