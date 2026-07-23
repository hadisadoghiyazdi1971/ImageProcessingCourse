from __future__ import annotations
import math
import torch
from torch import nn

class Coupling(nn.Module):
    def __init__(self, flip: bool = False) -> None:
        super().__init__(); self.flip = flip
        self.st = nn.Sequential(nn.Linear(1,32), nn.Tanh(), nn.Linear(32,32), nn.Tanh(), nn.Linear(32,2))
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a,b = (x[:,1:2],x[:,0:1]) if self.flip else (x[:,0:1],x[:,1:2])
        s,t = self.st(a).chunk(2,1); s = .8*torch.tanh(s)
        yb = b*torch.exp(s)+t
        y = torch.cat((yb,a),1) if self.flip else torch.cat((a,yb),1)
        return y, s.squeeze(1)
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        a,b = (y[:,1:2],y[:,0:1]) if self.flip else (y[:,0:1],y[:,1:2])
        s,t = self.st(a).chunk(2,1); s=.8*torch.tanh(s)
        xb=(b-t)*torch.exp(-s)
        return torch.cat((xb,a),1) if self.flip else torch.cat((a,xb),1)

class Flow(nn.Module):
    def __init__(self): super().__init__(); self.layers=nn.ModuleList([Coupling(False),Coupling(True),Coupling(False),Coupling(True)])
    def forward(self,x):
        logdet=torch.zeros(x.shape[0]); z=x
        for layer in self.layers:
            z,ld=layer(z); logdet=logdet+ld
        return z,logdet
    def inverse(self,z):
        x=z
        for layer in reversed(self.layers): x=layer.inverse(x)
        return x
    def nll(self,x):
        z,ld=self(x); logpz=-.5*(z.square().sum(1)+2*math.log(2*math.pi)); return -(logpz+ld).mean()

if __name__=='__main__':
    torch.manual_seed(4); f=Flow(); x=torch.randn(64,2)
    z,ld=f(x); xr=f.inverse(z)
    print({"reconstruction_max_error": (x-xr).abs().max().detach().item(), "mean_logdet": ld.mean().detach().item(), "nll": f.nll(x).detach().item()})
