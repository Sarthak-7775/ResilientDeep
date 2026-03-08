import torch
import torch.nn as nn
from torchvision import models

class VisibilityMatrix(nn.Module):
    """Identifies invisible compression artifacts through frequency domain analysis[cite: 176, 177]."""
    def __init__(self):
        super(VisibilityMatrix, self).__init__()
        # A simple 1x1 convolution to act as a trainable frequency filter
        self.filter = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        return self.filter(x)

class HighFreqEnhancer(nn.Module):
    """Neural network restores fine details before classification[cite: 178, 179]."""
    def __init__(self):
        super(HighFreqEnhancer, self).__init__()
        # Edge-enhancement simulation using a basic CNN block
        self.enhance = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        return x + self.enhance(x) # Residual connection

class ResilientDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(ResilientDetector, self).__init__()
        self.visibility_matrix = VisibilityMatrix()
        self.enhancer = HighFreqEnhancer()
        
        # Core Classification Backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        x = self.visibility_matrix(x)
        x = self.enhancer(x)
        return self.backbone(x)