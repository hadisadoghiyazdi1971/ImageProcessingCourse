from __future__ import annotations
import torch


def patchify(x: torch.Tensor, patch: int) -> torch.Tensor:
    """B,C,H,W -> B,N,(C*patch*patch). H,W must be divisible by patch."""
    if x.ndim != 4:
        raise ValueError('x must have shape [B,C,H,W]')
    b, c, h, w = x.shape
    if h % patch or w % patch:
        raise ValueError('height and width must be divisible by patch size')
    y = x.reshape(b, c, h // patch, patch, w // patch, patch)
    y = y.permute(0, 2, 4, 1, 3, 5).contiguous()
    return y.reshape(b, (h // patch) * (w // patch), c * patch * patch)


def unpatchify(tokens: torch.Tensor, patch: int, height: int, width: int, channels: int) -> torch.Tensor:
    """Inverse of patchify for a known image shape."""
    b, n, d = tokens.shape
    gh, gw = height // patch, width // patch
    if n != gh * gw or d != channels * patch * patch:
        raise ValueError('token shape is inconsistent with requested image shape')
    y = tokens.reshape(b, gh, gw, channels, patch, patch)
    y = y.permute(0, 3, 1, 4, 2, 5).contiguous()
    return y.reshape(b, channels, height, width)


if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 48)
    t = patchify(x, 8)
    xr = unpatchify(t, 8, 32, 48, 3)
    err = (x - xr).abs().max().item()
    print({'tokens': tuple(t.shape), 'max_reconstruction_error': err})
    assert err == 0.0
