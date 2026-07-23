from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from info_nce import nt_xent


class SmallEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.proj = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, out_dim))

    def forward(self, x):
        h = self.net(x)
        return h, self.proj(h)


def augment(x):
    # Educational tensor-only augmentation; use torchvision v2 for real training.
    if torch.rand(()) < .5:
        x = x.flip(-1)
    gain = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(.7, 1.3)
    noise = .04 * torch.randn_like(x)
    return (x * gain + noise).clamp(-3, 3)


if __name__ == '__main__':
    torch.manual_seed(3)
    model = SmallEncoder()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(32, 3, 32, 32)
    _, z1 = model(augment(x)); _, z2 = model(augment(x))
    loss = nt_xent(z1, z2, .2)
    opt.zero_grad(); loss.backward(); opt.step()
    print({'one_step_loss': float(loss.detach()), 'embedding_shape': tuple(z1.shape)})
    assert torch.isfinite(loss)
