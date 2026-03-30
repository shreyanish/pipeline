import torch
import torch.nn as nn

class DeepPhys(nn.Module):
    """
    DeepPhys model implementation based on Chen et al. (2018)
    Dual branch: Motion branch and Appearance branch
    """
    def __init__(self, in_channels=3, img_size=36):
        super(DeepPhys, self).__init__()
        
        # Appearance Branch
        self.appearance_conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.appearance_conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.appearance_avgpool = nn.AvgPool2d(2)
        self.appearance_dropout1 = nn.Dropout(0.25)
        
        self.appearance_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.appearance_conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.appearance_dropout2 = nn.Dropout(0.25)
        
        # Motion Branch
        self.motion_conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.motion_conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.motion_avgpool = nn.AvgPool2d(2)
        self.motion_dropout1 = nn.Dropout(0.25)
        
        self.motion_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.motion_conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.motion_dropout2 = nn.Dropout(0.25)
        
        # Attention modules (connecting Appearance to Motion)
        # DeepPhys uses appearance branch to guide motion branch
        self.attention1 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.attention2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final layers
        # After 2 sets of [conv, conv, pool], 36x36 becomes approx 16x16?
        # Let's calculate: 36 -> (padding1) 36 -> (kernel3) 34 -> (pool2) 17 -> (padding1) 17 -> (kernel3) 15
        # 15*15*64 = 14400
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 15 * 15, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x_appearance, x_motion):
        """
        x_appearance: (Batch, 3, H, W) - static frame information
        x_motion: (Batch, 3, H, W) - normalized frame difference
        """
        # Appearance Branch
        a1 = torch.relu(self.appearance_conv1(x_appearance))
        a2 = torch.relu(self.appearance_conv2(a1))
        
        # First Attention
        attn1 = self.attention1(a2)
        
        # Motion Branch 1
        m1 = torch.relu(self.motion_conv1(x_motion))
        m2 = torch.relu(self.motion_conv2(m1))
        m2_attended = m2 * attn1
        m3 = self.motion_avgpool(m2_attended)
        m3_pooled = self.motion_dropout1(m3)
        
        # Appearance skip
        a3 = self.appearance_avgpool(a2)
        a3_pooled = self.appearance_dropout1(a3)
        
        # Second stage
        a4 = torch.relu(self.appearance_conv3(a3_pooled))
        a5 = torch.relu(self.appearance_conv4(a4))
        
        # Second Attention
        attn2 = self.attention2(a5)
        
        m4 = torch.relu(self.motion_conv3(m3_pooled))
        m5 = torch.relu(self.motion_conv4(m4))
        m5_attended = m5 * attn2
        
        # Final layers
        out = self.flatten(m5_attended)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out

if __name__ == "__main__":
    model = DeepPhys()
    x_a = torch.randn(1, 3, 36, 36)
    x_m = torch.randn(1, 3, 36, 36)
    y = model(x_a, x_m)
    print(f"Output shape: {y.shape}")
