from __future__ import annotations
import numpy as np

def sqrtm_psd(a: np.ndarray) -> np.ndarray:
    w,v=np.linalg.eigh((a+a.T)/2); return (v*np.sqrt(np.clip(w,0,None)))@v.T

def fid(x: np.ndarray,y: np.ndarray) -> float:
    mx,my=x.mean(0),y.mean(0); cx=np.cov(x,rowvar=False); cy=np.cov(y,rowvar=False)
    cross=sqrtm_psd(sqrtm_psd(cx)@cy@sqrtm_psd(cx))
    return float(np.sum((mx-my)**2)+np.trace(cx+cy-2*cross))

def kid_unbiased(x: np.ndarray,y: np.ndarray) -> float:
    d=x.shape[1]
    kxx=(x@x.T/d+1)**3; kyy=(y@y.T/d+1)**3; kxy=(x@y.T/d+1)**3
    m,n=len(x),len(y)
    return float((kxx.sum()-np.trace(kxx))/(m*(m-1))+(kyy.sum()-np.trace(kyy))/(n*(n-1))-2*kxy.mean())

if __name__=='__main__':
    rng=np.random.default_rng(0); real=rng.normal(size=(500,64)); good=rng.normal(.05,1.03,size=(500,64)); bad=rng.normal(.5,1.4,size=(500,64))
    print({"fid_good":fid(real,good),"fid_bad":fid(real,bad),"kid_good":kid_unbiased(real,good),"kid_bad":kid_unbiased(real,bad)})
