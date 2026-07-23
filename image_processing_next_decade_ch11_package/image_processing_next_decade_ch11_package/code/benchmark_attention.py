from __future__ import annotations
import argparse, time, json
import torch
from torch.nn import functional as F


def bench(n=256, d=64, heads=4, iters=20):
    q=torch.randn(1,heads,n,d)
    k=torch.randn_like(q); v=torch.randn_like(q)
    for _ in range(3): F.scaled_dot_product_attention(q,k,v)
    times=[]
    for _ in range(iters):
        t=time.perf_counter(); F.scaled_dot_product_attention(q,k,v); times.append(time.perf_counter()-t)
    return {'tokens':n,'head_dim':d,'heads':heads,'median_ms':1000*float(torch.tensor(times).median()),
            'backend':'cpu' if not torch.cuda.is_available() else torch.cuda.get_device_name(0)}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--tokens',type=int,default=128); args=p.parse_args()
    result=bench(n=args.tokens,iters=5); print(json.dumps(result,indent=2))
