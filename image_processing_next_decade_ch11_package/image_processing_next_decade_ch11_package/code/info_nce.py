from __future__ import annotations
import torch
from torch.nn import functional as F


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Symmetric InfoNCE / NT-Xent for two aligned batches."""
    if z1.shape != z2.shape or z1.ndim != 2:
        raise ValueError('z1 and z2 must have equal shape [B,D]')
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    logits12 = z1 @ z2.T / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)
    return 0.5 * (F.cross_entropy(logits12, labels) + F.cross_entropy(logits12.T, labels))


if __name__ == '__main__':
    torch.manual_seed(2)
    base = torch.randn(16, 64)
    good = nt_xent(base + .02*torch.randn_like(base), base + .02*torch.randn_like(base))
    bad = nt_xent(torch.randn_like(base), torch.randn_like(base))
    print({'aligned_loss': float(good), 'random_loss': float(bad)})
    assert good < bad
