from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class PackedBatch:
    tokens: torch.Tensor
    segment_ids: torch.Tensor
    lengths: list[int]


def pack_sequences(sequences: list[torch.Tensor]) -> PackedBatch:
    if not sequences or any(x.ndim != 2 for x in sequences):
        raise ValueError('provide a non-empty list of [N_i,D] tensors')
    d = sequences[0].shape[1]
    if any(x.shape[1] != d for x in sequences):
        raise ValueError('embedding dimensions must match')
    tokens = torch.cat(sequences, dim=0)
    ids = torch.cat([torch.full((x.shape[0],), i, dtype=torch.long, device=x.device)
                     for i, x in enumerate(sequences)])
    return PackedBatch(tokens, ids, [x.shape[0] for x in sequences])


def block_diagonal_mask(segment_ids: torch.Tensor) -> torch.Tensor:
    """True means attention is permitted within the same packed image."""
    return segment_ids[:, None] == segment_ids[None, :]


if __name__ == '__main__':
    torch.manual_seed(6)
    seqs=[torch.randn(n,32) for n in [12,7,20,5]]
    packed=pack_sequences(seqs); mask=block_diagonal_mask(packed.segment_ids)
    print({'packed':tuple(packed.tokens.shape),'lengths':packed.lengths,'allowed_fraction':float(mask.float().mean())})
    assert mask.shape[0] == sum(packed.lengths)
