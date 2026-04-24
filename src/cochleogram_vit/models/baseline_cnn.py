import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    """
    A simple Convolutional Neural Network baseline.
    Expects input shape: (Batch, 1, 128, 128)
    Outputs logits: (Batch, num_classes)
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 4):
        super().__init__()
        
        self.features = nn.Sequential(
            # First Convolutional Layer
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 64x64
            
            # Second Convolutional Layer
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 32x32
        )
        
        # Deep Neural Network (DNN) block
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
