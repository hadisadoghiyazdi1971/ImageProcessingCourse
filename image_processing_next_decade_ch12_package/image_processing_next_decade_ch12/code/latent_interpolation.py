from __future__ import annotations
import torch

def lerp(a: torch.Tensor,b: torch.Tensor,t: torch.Tensor) -> torch.Tensor: return (1-t)*a+t*b

def slerp(a: torch.Tensor,b: torch.Tensor,t: torch.Tensor,eps: float=1e-7) -> torch.Tensor:
    an=a/a.norm(); bn=b/b.norm(); omega=torch.acos((an@bn).clamp(-1+eps,1-eps)); so=torch.sin(omega)
    if float(so.abs())<eps: return lerp(a,b,t)
    return torch.sin((1-t)*omega)/so*a + torch.sin(t*omega)/so*b

if __name__=='__main__':
    torch.manual_seed(0); a=torch.randn(16); b=torch.randn(16)
    ts=torch.linspace(0,1,7)
    ln=torch.stack([lerp(a,b,t).norm() for t in ts]); sn=torch.stack([slerp(a,b,t).norm() for t in ts])
    print({"linear_norms":[round(float(v),3) for v in ln],"spherical_norms":[round(float(v),3) for v in sn]})
