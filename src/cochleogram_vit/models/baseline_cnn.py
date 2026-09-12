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
        
        # Match the Baseline CNN from the paper/table: two conv layers
        # with kernels (5x5, 3x3), two 2x2 max pools, LeakyReLU activations.
        # Keep the classifier small (one Dense(128) before output) so total
        # parameters are in the ~8M range as in the reference table.
        self.features = nn.Sequential(
            # First Convolutional Layer (5x5)
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 64x64

            # Second Convolutional Layer (3x3)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 32x32
        )

        # Lightweight DNN block matching the BaselineCNN spec
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
