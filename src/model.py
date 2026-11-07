import torch
import torch.nn as nn


class ContrastiveAligner(nn.Module):
    """Placeholder module layout for the ACL 2025 paper."""

    def __init__(self, vision_dim: int = 768, text_dim: int = 768, embed_dim: int = 256):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)

    def forward(self, vision_features, text_features):
        return self.vision_proj(vision_features), self.text_proj(text_features)
