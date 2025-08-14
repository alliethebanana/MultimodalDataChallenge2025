import torch
import torch.nn as nn
import math

class FourierLatLon(nn.Module):
    """
    Map lat/lon (degrees) to smooth Fourier features.
    n_freqs=4 -> 16 dims: 2 axes * 2 (sin,cos) * n_freqs.
    """
    def __init__(self, n_freqs: int = 4):
        super().__init__()
        self.n_freqs = n_freqs
        # fixed frequency multipliers 1,2,4,8 (good default)
        self.register_buffer("freqs", torch.tensor([2**k for k in range(n_freqs)], dtype=torch.float32))

    def forward(self, latlon: torch.Tensor) -> torch.Tensor:
        """
        latlon: (N, 2) in degrees; can contain NaNs for missing.
        Returns (N, 4*n_freqs) features. Unknown rows -> zeros.
        """
        # lat in [-90,90], lon in [-180,180]
        lat = latlon[..., 0]
        lon = latlon[..., 1]
        # mask unknowns
        mask_ok = (~torch.isnan(lat)) & (~torch.isnan(lon))
        # normalize to [-1,1]
        lat_n = (lat / 90.0).clamp(-1.0, 1.0)
        lon_n = (lon / 180.0).clamp(-1.0, 1.0)

        # broadcast freqs: (N, n_freqs)
        freqs = self.freqs.to(lat_n.device)[None, :]

        # angles (N, n_freqs)
        ang_lat = 2 * math.pi * (lat_n.unsqueeze(1) * freqs)
        ang_lon = 2 * math.pi * (lon_n.unsqueeze(1) * freqs)

        # sin/cos per axis, then concat: [sin_lat, cos_lat, sin_lon, cos_lon]
        sin_lat = torch.sin(ang_lat)
        cos_lat = torch.cos(ang_lat)
        sin_lon = torch.sin(ang_lon)
        cos_lon = torch.cos(ang_lon)
        feats = torch.cat([sin_lat, cos_lat, sin_lon, cos_lon], dim=-1)  # (N, 4*n_freqs)

        # zero rows where lat/lon unknown
        feats = feats * mask_ok.unsqueeze(1).float()
        return feats  # (N, 4*n_freqs)
