from __future__ import annotations
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__(); self.net = nn.Sequential(nn.Linear(in_dim,64), nn.LeakyReLU(.2), nn.Linear(64,64), nn.LeakyReLU(.2), nn.Linear(64,out_dim))
    def forward(self, x): return self.net(x)

def gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    eps = torch.rand(real.shape[0], 1, device=real.device)
    xhat = eps * real + (1 - eps) * fake
    xhat.requires_grad_(True)
    score = critic(xhat).sum()
    grad, = torch.autograd.grad(score, xhat, create_graph=True)
    return ((grad.flatten(1).norm(2, dim=1) - 1) ** 2).mean()

if __name__ == "__main__":
    torch.manual_seed(11)
    G, C = MLP(8,2), MLP(2,1)
    real = torch.randn(32,2) + torch.tensor([1.5,-.5])
    fake = G(torch.randn(32,8)).detach()
    gp = gradient_penalty(C, real, fake)
    wasserstein = C(fake).mean() - C(real).mean()
    loss = wasserstein + 10 * gp
    loss.backward()
    print({"critic_loss": loss.detach().item(), "gradient_penalty": gp.detach().item()})
