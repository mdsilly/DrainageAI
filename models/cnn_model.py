"""
CNN model for drainage pipe detection.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from .base_model import BaseModel


class UNetDecoder(nn.Module):
    """U-Net decoder for segmentation."""
    
    def __init__(self, in_channels, out_channels):
        super(UNetDecoder, self).__init__()
        
        # Upsampling layers
        self.up1 = nn.ConvTranspose2d(in_channels, 512, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final output layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x, skip_connections):
        """Forward pass with skip connections from encoder."""
        x = self.up1(x)
        x = torch.cat([x, skip_connections[0]], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, skip_connections[1]], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, skip_connections[2]], dim=1)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x, skip_connections[3]], dim=1)
        x = self.conv4(x)
        
        x = self.final(x)
        return x


class CNNModel(BaseModel):
    """CNN model for drainage pipe detection using ResNet backbone and U-Net decoder."""
    
    def __init__(self, num_classes=1, pretrained=True):
        super(CNNModel, self).__init__()
        
        # Load pretrained ResNet-50 as encoder
        self.encoder = models.resnet50(pretrained=pretrained)
        
        # Remove the final fully connected layer
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        
        # U-Net decoder
        self.decoder = UNetDecoder(2048, num_classes)
        
        # Activation function for final output
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        """Forward pass of the model."""
        # Store skip connections for U-Net
        skip_connections = []
        
        # Run through ResNet layers and store skip connections
        for i, layer in enumerate(self.encoder.children()):
            x = layer(x)
            
            # Store skip connections at appropriate layers
            if i == 4:  # After layer1
                skip_connections.append(x)
            elif i == 5:  # After layer2
                skip_connections.append(x)
            elif i == 6:  # After layer3
                skip_connections.append(x)
            elif i == 7:  # After layer4
                skip_connections.append(x)
        
        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]
        
        # Decode
        x = self.decoder(x, skip_connections)
        
        # Apply activation
        x = self.activation(x)
        
        return x
    
    def preprocess(self, data):
        """Preprocess input data before feeding to the model."""
        # Implement preprocessing for satellite imagery
        # This would include normalization, resizing, etc.
        # For MVP, we'll assume data is already preprocessed
        return data
    
    def postprocess(self, output):
        """Postprocess model output to generate final predictions."""
        # Convert probability map to binary mask
        binary_mask = (output > 0.5).float()
        
        # Additional postprocessing could include:
        # - Removing small isolated predictions
        # - Connecting nearby segments
        # - Vectorizing the results
        
        return binary_mask
