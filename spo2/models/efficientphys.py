import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    def __init__(self, in_channels):
        super(TemporalAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.sigmoid(self.conv(x))
        return x * attn

class EfficientPhys(nn.Module):
    """
    EfficientPhys model implementation.
    Lightweight 2D CNN with Temporal Attention.
    """
    def __init__(self, in_channels=3):
        super(EfficientPhys, self).__init__()
        
        # 2D CNN Backbone (highly optimized)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.attn2 = TemporalAttention(32)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.attn3 = TemporalAttention(64)
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        x: (Batch, 3, H, W)
        """
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.attn2(x)
        x = self.enc3(x)
        x = self.attn3(x)
        x = self.enc4(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = EfficientPhys()
    x = torch.randn(1, 3, 72, 72) # EfficientPhys often uses larger inputs but 72 is common
    y = model(x)
    print(f"Output shape: {y.shape}")
