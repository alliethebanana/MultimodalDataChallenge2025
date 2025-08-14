import math
import torch
import torch.nn as nn

class CyclicMonth(nn.Module):
    """Encode month index (1..12 or 0 for unknown) to 2D sin/cos."""

    def __init__(self):
        super().__init__()

    def forward(self, month_idx: torch.Tensor) -> torch.Tensor:
        # month_idx can be float/int; 0 = unknown -> returns [0, 0]
        month = month_idx.float()
        # clamp to [0,12]; treat 0 specially
        month = torch.clamp(month, 0, 12)
        ang = 2 * math.pi * (month / 12.0)
        sin = torch.sin(ang)
        cos = torch.cos(ang)
        # zero-out unknowns (month == 0)
        mask_unknown = (month == 0)
        sin = sin.masked_fill(mask_unknown, 0.0)
        cos = cos.masked_fill(mask_unknown, 0.0)
        return torch.stack([sin, cos], dim=-1)  # (N, 2)
