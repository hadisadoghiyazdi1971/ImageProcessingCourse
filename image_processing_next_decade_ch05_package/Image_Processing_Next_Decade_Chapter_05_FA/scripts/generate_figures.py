from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from scipy.fft import fft2, ifft2, fftshift
from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_tv_chambolle, denoise_nl_means, estimate_sigma, wiener
from skimage.util import random_noise

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
OUT = ROOT / "outputs"
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(1405)


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(FIG / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close()

# Base image
x = img_as_float(data.camera())[64:448, 64:448]

# 1) Degradation/restoration taxonomy
sigma = 0.08
y_noise = np.clip(x + rng.normal(0, sigma, x.shape), 0, 1)
psf = np.ones((11,11), dtype=float); psf /= psf.sum()
y_blur = signal.fftconvolve(x, psf, mode='same')
y_both = np.clip(y_blur + rng.normal(0, 0.03, x.shape), 0, 1)
mask = rng.random(x.shape) > 0.35
y_mask = x * mask
fig, ax = plt.subplots(2,3, figsize=(11,7))
items=[(x,'Clean image'),(y_noise,'Denoising: H=I'),(y_blur,'Deblurring: convolution'),
       (y_both,'Restoration: blur + noise'),(y_mask,'Inpainting: masking'),(ndimage.zoom(ndimage.zoom(x,0.25,order=1),4,order=1)[:384,:384],'Super-resolution: down/up')]
for a,(im,t) in zip(ax.ravel(),items):
    a.imshow(im,cmap='gray',vmin=0,vmax=1); a.set_title(t); a.axis('off')
savefig('degradation_taxonomy')

# 2) Wiener frequency-domain behavior
N=512
f=np.linspace(-0.5,0.5,N)
H=np.exp(-(f/0.08)**2)
for K in [1e-4,1e-3,1e-2,1e-1]:
    G=np.conj(H)/(np.abs(H)**2+K)
    plt.plot(f,np.abs(G),label=f'K={K:g}')
plt.xlabel('Normalized frequency'); plt.ylabel('|G(f)|'); plt.title('Wiener inverse response and regularization')
plt.legend(); plt.grid(alpha=.3)
savefig('wiener_frequency_response')

# 3) Shrinkage functions
t=np.linspace(-4,4,1000); lam=1.2
hard=t*(np.abs(t)>lam)
soft=np.sign(t)*np.maximum(np.abs(t)-lam,0)
firm=np.sign(t)*np.where(np.abs(t)<=lam,0,np.where(np.abs(t)>=2.5*lam,np.abs(t),(2.5*lam*(np.abs(t)-lam))/(1.5*lam)))
plt.plot(t,t,'--',label='Identity')
plt.plot(t,hard,label='Hard')
plt.plot(t,soft,label='Soft / proximal L1')
plt.plot(t,firm,label='Firm')
plt.axvline(lam,ls=':'); plt.axvline(-lam,ls=':')
plt.xlabel('Noisy coefficient'); plt.ylabel('Estimated coefficient'); plt.title('Coefficient shrinkage rules')
plt.legend(); plt.grid(alpha=.3)
savefig('shrinkage_rules')

# 4) Patch matrix and low-rank spectrum
patch=8; refs=[]
base=x[170:178,150:158]
for i in range(80):
    di,dj=rng.integers(-35,36,2)
    p=x[170+di:178+di,150+dj:158+dj]
    if p.shape==(8,8): refs.append(p.ravel())
Y=np.stack(refs,axis=1)
Y=Y+rng.normal(0,.04,Y.shape)
s=np.linalg.svd(Y-Y.mean(axis=1,keepdims=True),compute_uv=False)
plt.semilogy(np.arange(1,len(s)+1),s,'o-')
plt.xlabel('Singular-value index'); plt.ylabel('Singular value (log scale)'); plt.title('Approximate low rank of grouped similar patches')
plt.grid(alpha=.3)
savefig('patch_group_singular_values')

# 5) Variational geometry and staircasing
xx=np.linspace(0,1,300)
truth=np.piecewise(xx,[xx<.25,(xx>=.25)&(xx<.55),xx>=.55],[lambda z:.25+0*z,lambda z:.75+0.18*np.sin(18*z),lambda z:.45+0*z])
y=truth+rng.normal(0,.10,truth.shape)
# 1D Tikhonov solve
n=len(y); D=np.eye(n)-np.eye(n,k=1); D=D[:-1]
alpha=.9
u=np.linalg.solve(np.eye(n)+alpha*D.T@D,y)
# 1D TV via skimage
v=denoise_tv_chambolle(y,weight=.12,channel_axis=None)
plt.plot(xx,y,alpha=.35,label='Noisy')
plt.plot(xx,truth,lw=2,label='Truth')
plt.plot(xx,u,lw=2,label='Tikhonov')
plt.plot(xx,v,lw=2,label='TV')
plt.xlabel('Position'); plt.ylabel('Intensity'); plt.title('Quadratic smoothness versus total variation')
plt.legend(); plt.grid(alpha=.3)
savefig('tikhonov_vs_tv_1d')

# 6) Bias-variance / parameter sweep benchmark
rows=[]
for sig in [0.04,0.08,0.12]:
    y=np.clip(x+rng.normal(0,sig,x.shape),0,1)
    for name,fun,params in [
        ('Gaussian',lambda z,p: ndimage.gaussian_filter(z,p),[.3,.6,1.0,1.5,2.2]),
        ('TV',lambda z,p: denoise_tv_chambolle(z,weight=p,channel_axis=None),[.01,.03,.06,.1,.18]),
        ('NLM',lambda z,p: denoise_nl_means(z,h=p*sig,patch_size=5,patch_distance=6,fast_mode=True,channel_axis=None),[.5,.7,.9,1.1,1.4])]:
        for p in params:
            z=np.clip(fun(y,p),0,1)
            rows.append({'noise_sigma':sig,'method':name,'parameter':p,
                         'psnr':peak_signal_noise_ratio(x,z,data_range=1),
                         'ssim':structural_similarity(x,z,data_range=1)})
df=pd.DataFrame(rows); df.to_csv(OUT/'denoising_parameter_sweep.csv',index=False)
fig,ax=plt.subplots(1,2,figsize=(10,4))
for name,g in df[df.noise_sigma==.08].groupby('method'):
    ax[0].plot(g.parameter,g.psnr,'o-',label=name)
    ax[1].plot(g.parameter,g.ssim,'o-',label=name)
ax[0].set_xlabel('Method parameter'); ax[0].set_ylabel('PSNR (dB)'); ax[0].set_title('Parameter sensitivity')
ax[1].set_xlabel('Method parameter'); ax[1].set_ylabel('SSIM'); ax[1].set_title('Metric-dependent optimum')
for a in ax: a.grid(alpha=.3); a.legend()
savefig('parameter_sensitivity')

# 7) Comparison panel
sig=.08; y=np.clip(x+rng.normal(0,sig,x.shape),0,1)
methods={
    'Noisy': y,
    'Gaussian': ndimage.gaussian_filter(y,1.0),
    'Median': ndimage.median_filter(y,size=3),
    'TV': denoise_tv_chambolle(y,weight=.08,channel_axis=None),
    'NLM': denoise_nl_means(y,h=.9*sig,patch_size=5,patch_distance=7,fast_mode=True,channel_axis=None),
}
fig,ax=plt.subplots(2,3,figsize=(11,7))
for a,(name,z) in zip(ax.ravel(), [('Clean',x),*methods.items()]):
    a.imshow(z,cmap='gray',vmin=0,vmax=1)
    if name=='Clean': title=name
    else: title=f"{name}\nPSNR={peak_signal_noise_ratio(x,z,data_range=1):.2f}, SSIM={structural_similarity(x,z,data_range=1):.3f}"
    a.set_title(title); a.axis('off')
savefig('classical_denoising_comparison')

# 8) Deblurring regularization path
psf=np.ones((9,9)); psf/=psf.sum(); yb=signal.fftconvolve(x,psf,mode='same'); yb=np.clip(yb+rng.normal(0,.01,x.shape),0,1)
regs=[1e-4,1e-3,1e-2,1e-1]
fig,ax=plt.subplots(1,5,figsize=(15,3.4))
ax[0].imshow(yb,cmap='gray',vmin=0,vmax=1); ax[0].set_title('Blurred/noisy'); ax[0].axis('off')
for a,r in zip(ax[1:],regs):
    z=np.clip(wiener(yb,psf,balance=r,clip=False),0,1)
    a.imshow(z,cmap='gray',vmin=0,vmax=1); a.set_title(f'balance={r:g}\nPSNR={peak_signal_noise_ratio(x,z,data_range=1):.2f}'); a.axis('off')
savefig('deblurring_regularization_path')

# 9) Proximal-gradient convergence on a synthetic quadratic-L1 problem
m,n=120,240
A=rng.normal(size=(m,n))/np.sqrt(m)
x0=np.zeros(n); idx=rng.choice(n,18,replace=False); x0[idx]=rng.normal(size=18)
y=A@x0+rng.normal(0,.02,m); L=np.linalg.norm(A,2)**2; lam=.04
softth=lambda u,t: np.sign(u)*np.maximum(np.abs(u)-t,0)
xi=np.zeros(n); xf=np.zeros(n); z=xf.copy(); tk=1.; oi=[]; of=[]
def obj(q): return .5*np.linalg.norm(A@q-y)**2+lam*np.linalg.norm(q,1)
for k in range(150):
    xi=softth(xi-(A.T@(A@xi-y))/L,lam/L); oi.append(obj(xi))
    xn=softth(z-(A.T@(A@z-y))/L,lam/L); tn=(1+np.sqrt(1+4*tk*tk))/2; z=xn+((tk-1)/tn)*(xn-xf); xf=xn; tk=tn; of.append(obj(xf))
plt.semilogy(np.array(oi)-min(of[-20:])+1e-12,label='ISTA')
plt.semilogy(np.array(of)-min(of[-20:])+1e-12,label='FISTA')
plt.xlabel('Iteration'); plt.ylabel('Objective gap (log)'); plt.title('Acceleration in proximal optimization')
plt.legend(); plt.grid(alpha=.3)
savefig('ista_fista_convergence')

# 10) Perception-distortion conceptual curve
d=np.linspace(.1,1,200); p=.12/(d+.03)+.03*d
plt.plot(d,p,lw=2)
pts=[(.2,.12/(.23)+.006,'Perceptual / generative'),(.55,.12/.58+.0165,'Balanced'),(.9,.12/.93+.027,'Distortion-oriented')]
for xx0,yy0,lab in pts:
    plt.scatter([xx0],[yy0]); plt.annotate(lab,(xx0,yy0),xytext=(6,8),textcoords='offset points')
plt.xlabel('Distortion (lower is better)'); plt.ylabel('Perceptual discrepancy (lower is better)'); plt.title('Conceptual perception-distortion trade-off')
plt.grid(alpha=.3)
savefig('perception_distortion_tradeoff')

# summary metrics
best=df.sort_values('psnr',ascending=False).groupby(['noise_sigma','method'],as_index=False).first()
best.to_csv(OUT/'best_classical_results.csv',index=False)
print(best.to_string(index=False))
