import torch
import torch.nn as nn
import torch.nn.functional as F

class SpO2Encoder(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model
        # Projector for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return self.projector(features)

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        # z_i, z_j: (Batch, Dim)
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)
        sim = torch.matmul(z, z.T) / self.temperature
        
        # Mask for positives
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        mask = torch.eye(labels.shape[0], device=z.device).bool()
        labels = labels[~mask].view(labels.shape[0], -1)
        sim = sim[~mask].view(sim.shape[0], -1)
        
        # Positive samples are at index batch_size-1 in the shifted sim matrix
        positives = sim[labels.bool()].view(labels.shape[0], -1)
        negatives = sim[~labels.bool()].view(sim.shape[0], -1)
        
        logits = torch.cat([positives, negatives], dim=1)
        loss = F.cross_entropy(logits, torch.zeros(logits.shape[0], dtype=torch.long, device=z.device))
        return loss

if __name__ == "__main__":
    from deepphys import DeepPhys
    base = DeepPhys()
    # Mocking a model that returns 128 features instead of 1
    # For now just testing the contrastive loss logic
    loss_fn = ContrastiveLoss()
    z1 = torch.randn(4, 32)
    z2 = torch.randn(4, 32)
    loss = loss_fn(z1, z2)
    print(f"Contrastive loss: {loss.item()}")
