from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class TinyVAE(nn.Module):
    def __init__(self, x_dim: int = 64, z_dim: int = 8) -> None:
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(x_dim, 48), nn.SiLU(), nn.Linear(48, 32), nn.SiLU())
        self.mu = nn.Linear(32, z_dim)
        self.logvar = nn.Linear(32, z_dim)
        self.dec = nn.Sequential(nn.Linear(z_dim, 32), nn.SiLU(), nn.Linear(32, 48), nn.SiLU(), nn.Linear(48, x_dim))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        return self.mu(h), self.logvar(h).clamp(-12, 8)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar

def elbo_loss(x: torch.Tensor, logits: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> tuple[torch.Tensor, dict[str, float]]:
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / x.shape[0]
    kl = -0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp()) / x.shape[0]
    loss = recon + beta * kl
    return loss, {"loss": loss.detach().item(), "recon": recon.detach().item(), "kl": kl.detach().item()}

if __name__ == "__main__":
    torch.manual_seed(7)
    model = TinyVAE()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    x = torch.bernoulli(torch.full((32, 64), 0.35))
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        logits, mu, logvar = model(x)
        loss, metrics = elbo_loss(x, logits, mu, logvar)
        loss.backward()
        opt.step()
    print(metrics)
