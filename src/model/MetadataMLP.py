import torch
import torch.nn as nn

class MetadataMLP(nn.Module):
    """
    Projects metadata features to target_dim with LayerNorm, Dropout and a small MLP.
    Uses a residual connection when input_dim == target_dim.
    """
    def __init__(self, in_dim: int, target_dim: int, hidden: int = 512, p_drop: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.drop_in = nn.Dropout(p_drop)
        self.ff = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, target_dim)
        )
        self.use_res = (in_dim == target_dim)

    def forward(self, x):
        xn = self.drop_in(self.norm(x))
        out = self.ff(xn)
        if self.use_res:
            out = out + xn
        return out