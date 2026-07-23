from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, z_dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, 32), nn.LeakyReLU(0.2), nn.Linear(32, 32), nn.LeakyReLU(0.2), nn.Linear(32, 2))
    def forward(self, z: torch.Tensor) -> torch.Tensor: return self.net(z)

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.LeakyReLU(0.2), nn.Linear(32, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1))
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.net(x).squeeze(-1)

def sample_real(n: int) -> torch.Tensor:
    k = torch.randint(0, 8, (n,))
    angle = 2 * torch.pi * k.float() / 8
    centers = torch.stack((2 * torch.cos(angle), 2 * torch.sin(angle)), 1)
    return centers + 0.08 * torch.randn(n, 2)

if __name__ == "__main__":
    torch.manual_seed(1)
    G, D = Generator(), Discriminator()
    og = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.9)); od = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.9))
    for _ in range(20):
        real = sample_real(128); fake = G(torch.randn(128, 8)).detach()
        od.zero_grad(set_to_none=True)
        dloss = F.softplus(-D(real)).mean() + F.softplus(D(fake)).mean(); dloss.backward(); od.step()
        og.zero_grad(set_to_none=True)
        fake = G(torch.randn(128, 8)); gloss = F.softplus(-D(fake)).mean(); gloss.backward(); og.step()
    print({"d_loss": dloss.detach().item(), "g_loss": gloss.detach().item(), "sample_std": G(torch.randn(512,8)).std(0).detach().tolist()})
