from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if mask_type not in {"A","B"}: raise ValueError("mask_type must be A or B")
        mask = torch.ones_like(self.weight)
        h,w = self.kernel_size; yc,xc=h//2,w//2
        mask[:,:,yc+1:,:]=0; mask[:,:,yc,xc+(mask_type=="B"):]=0
        self.register_buffer("mask",mask)
    def forward(self,x): return F.conv2d(x,self.weight*self.mask,self.bias,self.stride,self.padding,self.dilation,self.groups)

class TinyPixelCNN(nn.Module):
    def __init__(self, levels: int=16):
        super().__init__(); self.net=nn.Sequential(MaskedConv2d('A',1,32,7,padding=3),nn.ReLU(),MaskedConv2d('B',32,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,levels,1))
    def forward(self,x): return self.net(x)

if __name__=='__main__':
    torch.manual_seed(5); model=TinyPixelCNN(); x=torch.rand(2,1,8,8); y=model(x)
    x2=x.clone(); x2[:,:,7,7]=1-x2[:,:,7,7]
    delta=(model(x2)[:,:,:7,:]-y[:,:,:7,:]).abs().max()
    print({"output_shape": list(y.shape), "causality_test_max_delta": delta.detach().item()})
