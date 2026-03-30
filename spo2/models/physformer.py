import torch
import torch.nn as nn
import torch.nn.functional as F

class TubeletEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=192, patch_size=(4, 4, 4)):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (Batch, C, T, H, W)
        x = self.conv(x) # (B, D, T', H', W')
        x = x.flatten(2).transpose(1, 2) # (B, T'*H'*W', D)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, N, D)
        residual = x
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        return x + residual

class PhysFormer(nn.Module):
    """
    PhysFormer: Video-based rPPG measurement with Transformers.
    Simplified version focusing on SpO2 estimation.
    """
    def __init__(self, in_channels=3, embed_dim=192, num_heads=3, depth=6):
        super().__init__()
        
        self.patch_embed = TubeletEmbedding(in_channels, embed_dim)
        
        self.transformer_layers = nn.ModuleList([
            nn.Sequential(
                MultiHeadAttention(embed_dim, num_heads),
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim)
            )
            for _ in range(depth)
        ])
        
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, x):
        """
        x: (Batch, 3, T, H, W)
        """
        x = self.patch_embed(x) # (B, N, D)
        
        for layer in self.transformer_layers:
            # Simple residual structure for the MLP part
            residual = x
            x = layer[0](x) # Attention
            x = layer[1:](x) # Norm + MLP
            x = x + residual
            
        x = x.mean(dim=1) # Global average pooling over tokens
        x = self.head(x)
        return x

if __name__ == "__main__":
    # Example input: 1 batch, 3 channels, 32 frames, 64x64 resolution
    model = PhysFormer(depth=2) # Small depth for testing
    x = torch.randn(1, 3, 32, 64, 64)
    y = model(x)
    print(f"Output shape: {y.shape}")
