from __future__ import annotations
import torch
from torch import nn

class Velocity(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(3,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,2))
    def forward(self,x,t): return self.net(torch.cat((x,t),1))

def sample_pair(n: int):
    x0=torch.randn(n,2)
    k=torch.randint(0,8,(n,)); a=2*torch.pi*k.float()/8
    x1=torch.stack((2*torch.cos(a),2*torch.sin(a)),1)+.08*torch.randn(n,2)
    return x0,x1

if __name__=='__main__':
    torch.manual_seed(12); v=Velocity(); opt=torch.optim.Adam(v.parameters(),1e-3)
    for _ in range(10):
        x0,x1=sample_pair(256); t=torch.rand(256,1); xt=(1-t)*x0+t*x1; target=x1-x0
        loss=(v(xt,t)-target).square().mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    print({"flow_matching_loss": loss.detach().item()})
