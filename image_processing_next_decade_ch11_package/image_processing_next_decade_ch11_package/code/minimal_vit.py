from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class PatchEmbed(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_chans=3, dim=192):
        super().__init__()
        if image_size % patch_size:
            raise ValueError('image_size must be divisible by patch_size')
        self.grid = image_size // patch_size
        self.num_patches = self.grid ** 2
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 6, dropout: float = 0.0):
        super().__init__()
        if dim % heads:
            raise ValueError('dim must be divisible by heads')
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).reshape(b, n, d)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(hidden, dim), nn.Dropout(drop))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyViT(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_classes=10, dim=192, depth=4, heads=6):
        super().__init__()
        self.patch = PatchEmbed(image_size, patch_size, 3, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, self.patch.num_patches + 1, dim))
        self.blocks = nn.Sequential(*[Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward_features(self, x):
        x = self.patch(x)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        return self.norm(self.blocks(x))

    def forward(self, x):
        return self.head(self.forward_features(x)[:, 0])


if __name__ == '__main__':
    torch.manual_seed(1)
    model = TinyViT()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    loss = y.square().mean()
    loss.backward()
    n_params = sum(p.numel() for p in model.parameters())
    print({'output': tuple(y.shape), 'parameters': n_params, 'loss': float(loss.detach())})
    assert y.shape == (2, 10)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
