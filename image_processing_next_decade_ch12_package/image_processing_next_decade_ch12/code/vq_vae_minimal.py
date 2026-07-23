from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, codes: int = 32, dim: int = 8, beta: float = 0.25) -> None:
        super().__init__()
        self.codebook = nn.Embedding(codes, dim)
        nn.init.uniform_(self.codebook.weight, -1 / codes, 1 / codes)
        self.beta = beta

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = z_e.reshape(-1, z_e.shape[-1])
        dist = flat.square().sum(1, keepdim=True) - 2 * flat @ self.codebook.weight.T + self.codebook.weight.square().sum(1)
        ids = dist.argmin(1)
        z_q = self.codebook(ids).view_as(z_e)
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commit_loss = F.mse_loss(z_e, z_q.detach())
        loss = codebook_loss + self.beta * commit_loss
        z_st = z_e + (z_q - z_e).detach()
        probs = torch.bincount(ids, minlength=self.codebook.num_embeddings).float()
        probs = probs / probs.sum().clamp_min(1)
        perplexity = torch.exp(-(probs * (probs + 1e-12).log()).sum())
        return z_st, loss, ids.view(z_e.shape[:-1]), perplexity

class TinyVQVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(64, 48), nn.SiLU(), nn.Linear(48, 4 * 8))
        self.vq = VectorQuantizer(32, 8)
        self.dec = nn.Sequential(nn.Linear(4 * 8, 48), nn.SiLU(), nn.Linear(48, 64))

    def forward(self, x: torch.Tensor):
        z_e = self.enc(x).view(x.shape[0], 4, 8)
        z_q, qloss, ids, perplexity = self.vq(z_e)
        logits = self.dec(z_q.flatten(1))
        return logits, qloss, ids, perplexity

if __name__ == "__main__":
    torch.manual_seed(3)
    model = TinyVQVAE()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    x = torch.rand(24, 64)
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        logits, qloss, ids, perplexity = model(x)
        loss = F.mse_loss(torch.sigmoid(logits), x) + qloss
        loss.backward(); opt.step()
    print({"loss": loss.detach().item(), "perplexity": perplexity.detach().item(), "used_codes": int(ids.unique().numel())})
