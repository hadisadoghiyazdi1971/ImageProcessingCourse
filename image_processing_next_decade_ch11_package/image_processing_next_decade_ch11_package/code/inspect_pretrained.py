"""Optional online example. It does not run during the offline smoke test."""
from __future__ import annotations
import torch


def main():
    try:
        import timm
    except ImportError as exc:
        raise SystemExit('Install timm first: pip install timm') from exc
    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=0)
    model.eval()
    x=torch.randn(1,3,224,224)
    with torch.no_grad(): y=model(x)
    print({'model':model.__class__.__name__,'feature_shape':tuple(y.shape)})

if __name__=='__main__': main()
