from __future__ import annotations
import torch
from torch import nn
from patchify import patchify


def random_mask(tokens: torch.Tensor, ratio: float):
    b, n, d = tokens.shape
    keep = max(1, int(n * (1 - ratio)))
    noise = torch.rand(b, n, device=tokens.device)
    ids_shuffle = noise.argsort(dim=1)
    ids_restore = ids_shuffle.argsort(dim=1)
    ids_keep = ids_shuffle[:, :keep]
    visible = torch.gather(tokens, 1, ids_keep[..., None].expand(-1, -1, d))
    mask = torch.ones(b, n, device=tokens.device)
    mask[:, :keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    return visible, mask, ids_restore


class TinyMAE(nn.Module):
    def __init__(self, patch=4, dim=96, decoder_dim=64):
        super().__init__()
        self.patch = patch
        patch_dim = 3 * patch * patch
        self.embed = nn.Linear(patch_dim, dim)
        layer = nn.TransformerEncoderLayer(dim, 4, 4*dim, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.to_dec = nn.Linear(dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        dec = nn.TransformerEncoderLayer(decoder_dim, 4, 4*decoder_dim, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(dec, 1)
        self.pred = nn.Linear(decoder_dim, patch_dim)

    def forward(self, x, ratio=.75):
        target = patchify(x, self.patch)
        tokens = self.embed(target)
        visible, mask, restore = random_mask(tokens, ratio)
        encoded = self.encoder(visible)
        dec_vis = self.to_dec(encoded)
        b, n, _ = target.shape
        missing = n - dec_vis.shape[1]
        seq = torch.cat([dec_vis, self.mask_token.expand(b, missing, -1)], dim=1)
        seq = torch.gather(seq, 1, restore[..., None].expand(-1, -1, seq.shape[-1]))
        pred = self.pred(self.decoder(seq))
        loss = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum().clamp_min(1)
        return loss, pred, mask


if __name__ == '__main__':
    torch.manual_seed(4)
    model = TinyMAE()
    x = torch.randn(8, 3, 32, 32)
    loss, pred, mask = model(x)
    loss.backward()
    print({'loss': float(loss.detach()), 'prediction': tuple(pred.shape), 'masked_fraction': float(mask.mean().detach())})
    assert torch.isfinite(loss)
