from __future__ import annotations

import csv
import math
import time
import multiprocessing as mp
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, signal
from skimage import color, data, metrics, transform, util

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(1405)


def savefig(name: str) -> None:
    plt.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close()


def normalize01(x: NDArray[np.floating]) -> NDArray[np.float64]:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.percentile(x, [0.5, 99.5])
    return np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1)


def gaussian_kernel(size: int, sigma: float) -> NDArray[np.float64]:
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax, indexing="xy")
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()


def pad_kernel_to_shape(k: NDArray[np.float64], shape: tuple[int, int]) -> NDArray[np.float64]:
    out = np.zeros(shape, dtype=np.float64)
    kh, kw = k.shape
    out[:kh, :kw] = k
    return np.roll(np.roll(out, -(kh // 2), axis=0), -(kw // 2), axis=1)


def fig_lti_eigenfunction() -> None:
    n = 256
    x = np.arange(n)
    freqs = [0.04, 0.12, 0.23]
    h = signal.windows.gaussian(41, std=5)
    h /= h.sum()
    fig, axes = plt.subplots(len(freqs), 2, figsize=(10, 7), constrained_layout=True)
    for r, f in enumerate(freqs):
        s = np.cos(2 * np.pi * f * x)
        y = signal.fftconvolve(s, h, mode="same")
        H = np.sum(h * np.exp(-1j * 2 * np.pi * f * np.arange(-(len(h)//2), len(h)//2 + 1)))
        axes[r, 0].plot(x, s, lw=1.2, label="input")
        axes[r, 0].plot(x, y, lw=1.2, label="filtered")
        axes[r, 0].set_xlim(60, 190)
        axes[r, 0].set_title(f"f={f:.2f} cycles/pixel")
        axes[r, 0].grid(alpha=.25)
        if r == 0:
            axes[r, 0].legend()
        axes[r, 1].bar([0, 1], [1.0, abs(H)])
        axes[r, 1].set_xticks([0, 1], ["input amplitude", "output amplitude"])
        axes[r, 1].set_ylim(0, 1.05)
        axes[r, 1].set_title(f"|H(f)|={abs(H):.3f}")
        axes[r, 1].grid(axis="y", alpha=.25)
    savefig("lti_sinusoid_eigenfunctions")


def fig_dft_diagonalization() -> float:
    n = 48
    h = np.zeros(n)
    h[:7] = np.array([1, 2, 4, 6, 4, 2, 1], dtype=float)
    h /= h.sum()
    C = np.zeros((n, n), dtype=complex)
    for j in range(n):
        C[:, j] = np.roll(h, j)
    k = np.arange(n)
    F = np.exp(-2j * np.pi * np.outer(k, k) / n) / np.sqrt(n)
    D = F @ C @ F.conj().T
    off = D - np.diag(np.diag(D))
    ratio = np.linalg.norm(off) / np.linalg.norm(D)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].imshow(np.abs(C), aspect="auto")
    axes[0].set_title("Circular convolution matrix |C|")
    axes[0].set_xlabel("input index")
    axes[0].set_ylabel("output index")
    axes[1].imshow(np.log10(np.abs(D) + 1e-16), aspect="auto")
    axes[1].set_title(r"$\log_{10}|FCF^*|$")
    axes[1].set_xlabel("frequency index")
    axes[1].set_ylabel("frequency index")
    savefig("dft_diagonalizes_circulant")
    return float(ratio)


def fig_phase_magnitude() -> dict[str, float]:
    img = util.img_as_float(data.camera())
    F = np.fft.fft2(img)
    mag = np.abs(F)
    phase = np.angle(F)
    phase_only = np.real(np.fft.ifft2(np.exp(1j * phase) * np.mean(mag)))
    mag_only = np.real(np.fft.ifft2(mag))
    shuffled_phase = RNG.permutation(phase.ravel()).reshape(phase.shape)
    shuffle_rec = np.real(np.fft.ifft2(mag * np.exp(1j * shuffled_phase)))
    figs = [img, normalize01(phase_only), normalize01(mag_only), normalize01(shuffle_rec)]
    titles = ["Real photograph", "Original phase + flat magnitude", "Original magnitude + zero phase", "Original magnitude + shuffled phase"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    for ax, im, title in zip(axes, figs, titles):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    savefig("phase_magnitude_real_image")
    return {
        "phase_only_ssim": float(metrics.structural_similarity(img, normalize01(phase_only), data_range=1)),
        "magnitude_only_ssim": float(metrics.structural_similarity(img, normalize01(mag_only), data_range=1)),
        "shuffled_phase_ssim": float(metrics.structural_similarity(img, normalize01(shuffle_rec), data_range=1)),
    }


def fig_boundary_conditions() -> dict[str, float]:
    img = util.img_as_float(data.camera())[40:360, 40:360]
    sigma = 10.0
    modes = ["constant", "wrap", "reflect"]
    outputs = [ndimage.gaussian_filter(img, sigma=sigma, mode=m) for m in modes]
    ref = ndimage.gaussian_filter(np.pad(img, 80, mode="reflect"), sigma=sigma, mode="reflect")[80:-80, 80:-80]
    border = np.zeros_like(img, dtype=bool)
    border[:30, :] = border[-30:, :] = True
    border[:, :30] = border[:, -30:] = True
    errors = {m: float(np.sqrt(np.mean((o[border] - ref[border])**2))) for m, o in zip(modes, outputs)}
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("Input crop")
    axes[0, 0].axis("off")
    for i, (m, o) in enumerate(zip(modes, outputs), start=1):
        axes[0, i].imshow(o, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"mode={m}")
        axes[0, i].axis("off")
    axes[1, 0].imshow(ref, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("large reflected reference")
    axes[1, 0].axis("off")
    for i, (m, o) in enumerate(zip(modes, outputs), start=1):
        axes[1, i].imshow(np.abs(o - ref), cmap="magma", vmin=0, vmax=0.15)
        axes[1, i].set_title(f"abs error; border RMSE={errors[m]:.4f}", fontsize=8)
        axes[1, i].axis("off")
    savefig("boundary_conditions_real_image")
    return errors


def fig_filter_responses() -> None:
    size = 65
    kernels = {
        "Box 9x9": np.pad(np.ones((9, 9))/81, ((28, 28), (28, 28))),
        "Gaussian sigma=3": gaussian_kernel(size, 3),
        "Laplacian": np.pad(np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=float), ((31,31),(31,31))),
        "Unsharp (a=1.5)": None,
    }
    g = gaussian_kernel(size, 2.5)
    delta = np.zeros((size, size)); delta[size//2, size//2] = 1
    kernels["Unsharp (a=1.5)"] = delta + 1.5*(delta-g)
    fig, axes = plt.subplots(2, len(kernels), figsize=(13, 6), constrained_layout=True)
    for c, (name, k) in enumerate(kernels.items()):
        axes[0,c].imshow(k, cmap="coolwarm")
        axes[0,c].set_title(name, fontsize=9)
        axes[0,c].axis("off")
        H = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(k), s=(256,256)))
        axes[1,c].imshow(np.log1p(np.abs(H)), cmap="viridis")
        axes[1,c].set_title("log(1+|H|)", fontsize=9)
        axes[1,c].axis("off")
    savefig("spatial_frequency_filter_pairs")


def fig_ringing() -> dict[str, float]:
    n = 1024
    x = np.linspace(-1, 1, n, endpoint=False)
    step = (x >= 0).astype(float)
    F = np.fft.fft(step)
    f = np.fft.fftfreq(n)
    cutoffs = [0.03, 0.08, 0.16]
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.plot(x, step, lw=2, label="ideal step")
    overs = {}
    for fc in cutoffs:
        mask = np.abs(f) <= fc
        rec = np.real(np.fft.ifft(F * mask))
        ax.plot(x, rec, lw=1.2, label=f"ideal LP fc={fc}")
        overs[str(fc)] = float(rec.max() - 1.0)
    ax.set_xlim(-0.18, 0.18)
    ax.set_ylim(-0.2, 1.2)
    ax.grid(alpha=.25)
    ax.legend()
    ax.set_title("Gibbs ringing from an ideal frequency cut-off")
    savefig("gibbs_ringing")
    return overs


def fig_scale_space() -> dict[str, float]:
    img = util.img_as_float(data.camera())
    sigmas = [0, 1, 2, 4, 8]
    fig, axes = plt.subplots(1, len(sigmas), figsize=(15, 3.3), constrained_layout=True)
    for ax, s in zip(axes, sigmas):
        out = img if s == 0 else ndimage.gaussian_filter(img, s, mode="reflect")
        ax.imshow(out, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"sigma={s}")
        ax.axis("off")
    savefig("gaussian_scale_space_real")
    s1, s2 = 1.7, 2.9
    lhs = ndimage.gaussian_filter(ndimage.gaussian_filter(img, s1, mode="reflect"), s2, mode="reflect")
    rhs = ndimage.gaussian_filter(img, math.sqrt(s1*s1+s2*s2), mode="reflect")
    rel = np.linalg.norm(lhs-rhs)/np.linalg.norm(rhs)
    # extrema count across scale for a 1D slice
    row = img[260]
    scales = np.geomspace(0.25, 12, 40)
    extrema=[]
    for s in scales:
        z=ndimage.gaussian_filter1d(row,s,mode='reflect')
        ext=np.count_nonzero(np.diff(np.sign(np.diff(z))))
        extrema.append(ext)
    fig, ax=plt.subplots(figsize=(7,4), constrained_layout=True)
    ax.semilogx(scales, extrema, marker='o', ms=3)
    ax.set_xlabel('Gaussian scale sigma')
    ax.set_ylabel('number of discrete extrema on one row')
    ax.grid(alpha=.25)
    ax.set_title('Empirical non-creation trend of extrema across scale')
    savefig('scale_space_extrema')
    return {"semigroup_relative_error": float(rel), "extrema_initial": int(extrema[0]), "extrema_final": int(extrema[-1])}


def resize_to(im: NDArray[np.float64], shape: tuple[int,int]) -> NDArray[np.float64]:
    return transform.resize(im, shape, order=1, mode="reflect", anti_aliasing=False, preserve_range=True)


def laplacian_pyramid(img: NDArray[np.float64], levels: int=4):
    gs=[img]
    for _ in range(levels):
        sm=ndimage.gaussian_filter(gs[-1],1.0,mode='reflect')
        gs.append(sm[::2,::2])
    ls=[]
    for i in range(levels):
        up=resize_to(gs[i+1],gs[i].shape)
        ls.append(gs[i]-up)
    ls.append(gs[-1])
    return gs,ls


def reconstruct_laplacian(ls):
    x=ls[-1]
    for level in reversed(ls[:-1]):
        x=resize_to(x,level.shape)+level
    return x


def fig_pyramids() -> dict[str,float]:
    img=util.img_as_float(data.camera())
    gs,ls=laplacian_pyramid(img,4)
    rec=reconstruct_laplacian(ls)
    fig,axes=plt.subplots(2,5,figsize=(14,6),constrained_layout=True)
    for i,g in enumerate(gs):
        axes[0,i].imshow(g,cmap='gray',vmin=0,vmax=1)
        axes[0,i].set_title(f'Gaussian {i}\n{g.shape[1]}x{g.shape[0]}',fontsize=8)
        axes[0,i].axis('off')
    for i,l in enumerate(ls):
        axes[1,i].imshow(normalize01(l),cmap='gray')
        axes[1,i].set_title(f'Laplacian {i}',fontsize=8)
        axes[1,i].axis('off')
    savefig('gaussian_laplacian_pyramids')
    return {"laplacian_reconstruction_max_abs":float(np.max(np.abs(rec-img))),"laplacian_reconstruction_rel_l2":float(np.linalg.norm(rec-img)/np.linalg.norm(img))}


def haar2(img: NDArray[np.float64]):
    a=img[0::2,0::2]; b=img[0::2,1::2]; c=img[1::2,0::2]; d=img[1::2,1::2]
    ll=(a+b+c+d)/2
    lh=(a-b+c-d)/2
    hl=(a+b-c-d)/2
    hh=(a-b-c+d)/2
    return ll,lh,hl,hh


def ihaar2(coeffs):
    ll,lh,hl,hh=coeffs
    a=(ll+lh+hl+hh)/2
    b=(ll-lh+hl-hh)/2
    c=(ll+lh-hl-hh)/2
    d=(ll-lh-hl+hh)/2
    out=np.empty((ll.shape[0]*2,ll.shape[1]*2),dtype=float)
    out[0::2,0::2]=a; out[0::2,1::2]=b; out[1::2,0::2]=c; out[1::2,1::2]=d
    return out


def fig_haar() -> dict[str,float]:
    img=util.img_as_float(data.camera())
    cs=haar2(img)
    rec=ihaar2(cs)
    fig,axes=plt.subplots(1,5,figsize=(15,3.2),constrained_layout=True)
    titles=['Input','LL','LH','HL','HH']
    ims=[img,cs[0],normalize01(cs[1]),normalize01(cs[2]),normalize01(cs[3])]
    for ax,im,t in zip(axes,ims,titles):
        ax.imshow(im,cmap='gray',vmin=0,vmax=1)
        ax.set_title(t)
        ax.axis('off')
    savefig('haar_wavelet_real_image')
    return {"haar_reconstruction_max_abs":float(np.max(np.abs(rec-img))),"haar_energy_ratio":float(sum(np.sum(c*c) for c in cs)/np.sum(img*img))}


def fig_gabor_bank() -> None:
    img=util.img_as_float(data.camera())
    img=transform.resize(img,(256,256),anti_aliasing=True)
    thetas=np.linspace(0,np.pi,6,endpoint=False)
    responses=[]
    for th in thetas:
        real,imag=ndimage.gabor_filter(img,frequency=0.18,theta=th,sigma_x=3,sigma_y=5) if hasattr(ndimage,'gabor_filter') else (None,None)
        if real is None:
            # Construct real Gabor kernel manually
            y,x=np.mgrid[-15:16,-15:16]
            xr=x*np.cos(th)+y*np.sin(th)
            yr=-x*np.sin(th)+y*np.cos(th)
            ker=np.exp(-(xr*xr/(2*3**2)+yr*yr/(2*5**2)))*np.cos(2*np.pi*0.18*xr)
            ker-=ker.mean()
            real=signal.fftconvolve(img,ker,mode='same')
            imag=np.zeros_like(real)
        responses.append(np.hypot(real,imag))
    fig,axes=plt.subplots(2,3,figsize=(10,7),constrained_layout=True)
    for ax,r,th in zip(axes.ravel(),responses,thetas):
        ax.imshow(r,cmap='magma')
        ax.set_title(f'theta={np.degrees(th):.0f} deg')
        ax.axis('off')
    savefig('gabor_orientation_bank')


def slanted_edge_mtf() -> dict[str,float]:
    n=512
    yy,xx=np.mgrid[0:n,0:n]
    theta=np.deg2rad(5.0)
    coord=(xx-n/2)*np.cos(theta)+(yy-n/2)*np.sin(theta)
    edge=(coord>=0).astype(float)
    sigma=1.15
    blurred=ndimage.gaussian_filter(edge,sigma,mode='reflect')
    noisy=np.clip(blurred+RNG.normal(0,0.002,blurred.shape),0,1)
    # Bin by distance to edge
    d=coord.ravel(); z=noisy.ravel()
    bins=np.linspace(-20,20,801)
    idx=np.digitize(d,bins)-1
    esf=np.zeros(len(bins)-1); cnt=np.zeros(len(bins)-1)
    np.add.at(esf,idx[(idx>=0)&(idx<len(esf))],z[(idx>=0)&(idx<len(esf))])
    np.add.at(cnt,idx[(idx>=0)&(idx<len(esf))],1)
    valid=cnt>0
    centers=0.5*(bins[:-1]+bins[1:])
    esf[valid]/=cnt[valid]
    esf=np.interp(centers,centers[valid],esf[valid])
    esf=ndimage.gaussian_filter1d(esf,1)
    dx=centers[1]-centers[0]
    lsf=np.gradient(esf,dx)
    lsf*=signal.windows.hann(len(lsf))
    mtf=np.abs(np.fft.rfft(lsf)); mtf/=mtf[0]
    freq=np.fft.rfftfreq(len(lsf),d=dx)
    theoretical=np.exp(-2*(np.pi*sigma*freq)**2)
    # MTF50 interpolation
    pos=np.where(mtf<=0.5)[0]
    mtf50=float(freq[pos[0]]) if len(pos) else float('nan')
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    axes[0].imshow(noisy,cmap='gray',vmin=0,vmax=1)
    axes[0].set_title('Controlled slanted edge')
    axes[0].axis('off')
    axes[1].plot(centers,esf,label='ESF')
    axes[1].plot(centers,lsf/np.max(np.abs(lsf)),label='normalized LSF')
    axes[1].set_xlim(-8,8); axes[1].grid(alpha=.25); axes[1].legend(); axes[1].set_title('ESF and LSF')
    axes[2].plot(freq,mtf,label='estimated MTF')
    axes[2].plot(freq,theoretical,'--',label='Gaussian theory')
    axes[2].axhline(.5,lw=.8); axes[2].set_xlim(0,.5); axes[2].set_ylim(0,1.05)
    axes[2].grid(alpha=.25); axes[2].legend(); axes[2].set_title(f'MTF50={mtf50:.3f} cyc/pixel')
    savefig('slanted_edge_mtf')
    return {"slanted_edge_mtf50":mtf50}


def fig_fft_complexity() -> dict[str,float]:
    ns=np.array([16,32,64,128,256,512])
    td=[]; tf=[]
    for n in ns:
        x=RNG.normal(size=n)+1j*RNG.normal(size=n)
        k=np.arange(n); W=np.exp(-2j*np.pi*np.outer(k,k)/n)
        t0=time.perf_counter();
        for _ in range(max(1,20000//(n*n))): y=W@x
        t1=time.perf_counter(); loops=max(1,20000//(n*n)); td.append((t1-t0)/loops)
        t0=time.perf_counter();
        loops2=5000
        for _ in range(loops2): y2=np.fft.fft(x)
        t1=time.perf_counter(); tf.append((t1-t0)/loops2)
    fig,ax=plt.subplots(figsize=(7,4.5),constrained_layout=True)
    ax.loglog(ns,td,'o-',label='direct matrix DFT')
    ax.loglog(ns,tf,'o-',label='FFT')
    ax.loglog(ns,td[0]*(ns/ns[0])**2,'--',label=r'$O(N^2)$ reference')
    ax.loglog(ns,tf[0]*(ns*np.log2(ns))/(ns[0]*np.log2(ns[0])),'--',label=r'$O(N\log N)$ reference')
    ax.set_xlabel('N'); ax.set_ylabel('seconds'); ax.grid(alpha=.25,which='both'); ax.legend(); ax.set_title('Measured complexity on the current CPU')
    savefig('fft_complexity_measured')
    return {"fft_speedup_n512":float(td[-1]/tf[-1])}


def fig_antialias_real() -> dict[str,float]:
    img=util.img_as_float(data.camera())
    # use 4x downsample; area/anti-aliased resize as reference
    ref=transform.resize(img,(128,128),order=1,anti_aliasing=True,preserve_range=True)
    naive=img[::4,::4]
    pre=ndimage.gaussian_filter(img,1.5,mode='reflect')[::4,::4]
    p_naive=metrics.peak_signal_noise_ratio(ref,naive,data_range=1)
    p_pre=metrics.peak_signal_noise_ratio(ref,pre,data_range=1)
    fig,axes=plt.subplots(1,4,figsize=(12,3.2),constrained_layout=True)
    ims=[ref,naive,pre,np.abs(naive-ref)]
    titles=['anti-aliased resize reference',f'naive decimation\nPSNR={p_naive:.2f} dB',f'prefilter + decimate\nPSNR={p_pre:.2f} dB','|naive-reference|']
    for ax,im,t in zip(axes,ims,titles):
        ax.imshow(im,cmap='gray',vmin=0,vmax=(.25 if '|' in t else 1))
        ax.set_title(t,fontsize=8); ax.axis('off')
    savefig('antialias_real_downsampling')
    return {"naive_downsample_psnr":float(p_naive),"prefilter_downsample_psnr":float(p_pre)}


def _fig_spectral_bias_impl() -> dict[str,float]:
    import torch
    torch.manual_seed(1405)
    torch.set_num_threads(1)
    x=torch.linspace(-1,1,256).unsqueeze(1)
    low=torch.sin(2*math.pi*x)
    high=0.35*torch.sin(20*math.pi*x)
    y=low+high
    class MLP(torch.nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net=torch.nn.Sequential(torch.nn.Linear(in_dim,32),torch.nn.Tanh(),torch.nn.Linear(32,32),torch.nn.Tanh(),torch.nn.Linear(32,1))
        def forward(self,z): return self.net(z)
    B=torch.arange(1,11,dtype=torch.float32).reshape(1,-1)*math.pi
    def feats(z): return torch.cat([torch.sin(z@B),torch.cos(z@B)],dim=1)
    models={'plain MLP':(MLP(1),lambda z:z),'Fourier-feature MLP':(MLP(20),feats)}
    records={}
    for name,(model,phi) in models.items():
        opt=torch.optim.Adam(model.parameters(),lr=1e-3)
        rec=[]
        for ep in range(401):
            pred=model(phi(x)); loss=((pred-y)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if ep in [0,2,5,10,20,50,100,200,400]:
                with torch.no_grad():
                    p=model(phi(x))
                    a1=float((p*low).sum()/(low*low).sum())
                    a10=float((p*torch.sin(20*math.pi*x)).sum()/(torch.sin(20*math.pi*x)**2).sum())
                    rec.append((ep,a1,a10,float(loss)))
        records[name]=rec
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    for name,rec in records.items():
        e=np.array([r[0] for r in rec]); a1=np.array([r[1] for r in rec]); a10=np.array([r[2] for r in rec])
        axes[0].semilogx(e+1,np.abs(a1),marker='o',label=name)
        axes[1].semilogx(e+1,np.abs(a10),marker='o',label=name)
    axes[0].axhline(1,ls='--',lw=.8); axes[1].axhline(.35,ls='--',lw=.8)
    axes[0].set_title('learned low-frequency amplitude')
    axes[1].set_title('learned high-frequency amplitude')
    for ax in axes:
        ax.set_xlabel('training epoch + 1'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    savefig('spectral_bias_fourier_features')
    plain_final=records['plain MLP'][-1]
    ff_final=records['Fourier-feature MLP'][-1]
    return {"plain_high_amp_final":plain_final[2],"fourier_high_amp_final":ff_final[2],"plain_loss_final":plain_final[3],"fourier_loss_final":ff_final[3]}


def _spectral_bias_worker(queue) -> None:
    try:
        queue.put((True, _fig_spectral_bias_impl()))
    except Exception as exc:  # pragma: no cover - propagated to parent
        queue.put((False, repr(exc)))


def fig_spectral_bias() -> dict[str, float]:
    """Run Torch in a spawned process to avoid BLAS/OpenMP interference."""
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_spectral_bias_worker, args=(queue,))
    process.start()
    process.join(timeout=120)
    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError("The spectral-bias experiment exceeded 120 seconds.")
    if queue.empty():
        raise RuntimeError(f"The spectral-bias worker exited with code {process.exitcode}.")
    ok, payload = queue.get()
    if not ok:
        raise RuntimeError(f"The spectral-bias worker failed: {payload}")
    return payload


def integrated_convolution_experiment() -> dict[str,float]:
    img=util.img_as_float(data.camera())
    k=gaussian_kernel(31,4.0)
    t0=time.perf_counter(); spatial=signal.convolve2d(img,k,mode='same',boundary='symm'); ts=time.perf_counter()-t0
    t0=time.perf_counter(); fft=signal.fftconvolve(img,k,mode='same'); tf=time.perf_counter()-t0
    # FFT zero-boundary is not same as symmetric; compare against zero-boundary spatial
    spatial_zero=signal.convolve2d(img,k,mode='same',boundary='fill',fillvalue=0)
    rel=np.linalg.norm(spatial_zero-fft)/np.linalg.norm(spatial_zero)
    # central region symmetry vs zero boundary should match closely
    center=(slice(40,-40),slice(40,-40))
    interior=np.linalg.norm(spatial[center]-fft[center])/np.linalg.norm(spatial[center])
    return {"spatial_symmetric_seconds":ts,"fft_zero_seconds":tf,"fft_vs_zero_spatial_rel_l2":float(rel),"symmetric_vs_zero_interior_rel_l2":float(interior)}



def latex_command_name(metric: str) -> str:
    """Return a TeX-safe command suffix (letters only)."""
    digit_words = {"0": "Zero", "1": "One", "2": "Two", "3": "Three",
                   "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
                   "8": "Eight", "9": "Nine"}
    parts = metric.replace('.', '_').split('_')
    safe_parts = []
    for part in parts:
        converted = ''.join(digit_words.get(ch, ch) for ch in part)
        safe_parts.append(converted[:1].upper() + converted[1:])
    return 'chthree' + ''.join(safe_parts)

def main() -> None:
    results = {}

    def run_step(label, fn):
        t0 = time.perf_counter()
        print(f"[start] {label}", flush=True)
        value = fn()
        print(f"[done ] {label}: {time.perf_counter() - t0:.2f} s", flush=True)
        return value

    # Run the Torch experiment first. On some Linux builds, initializing Torch
    # after a long SciPy/Matplotlib session can lead to an OpenMP deadlock.
    results.update(run_step('spectral bias', fig_spectral_bias))
    run_step('LSI eigenfunctions', fig_lti_eigenfunction)
    results['dft_offdiag_ratio'] = run_step('DFT diagonalization', fig_dft_diagonalization)
    results.update(run_step('phase and magnitude', fig_phase_magnitude))
    for key, value in run_step('boundary conditions', fig_boundary_conditions).items():
        results[f'boundary_{key}_rmse'] = value
    run_step('filter responses', fig_filter_responses)
    for key, value in run_step('Gibbs ringing', fig_ringing).items():
        results[f'gibbs_overshoot_fc_{key}'] = value
    results.update(run_step('Gaussian scale space', fig_scale_space))
    results.update(run_step('Gaussian/Laplacian pyramids', fig_pyramids))
    results.update(run_step('Haar wavelet', fig_haar))
    run_step('Gabor bank', fig_gabor_bank)
    results.update(run_step('slanted-edge MTF', slanted_edge_mtf))
    results.update(run_step('FFT complexity', fig_fft_complexity))
    results.update(run_step('anti-aliasing', fig_antialias_real))
    results.update(run_step('integrated convolution', integrated_convolution_experiment))

    with open(OUT.parent / 'experiment_results.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['metric', 'value'])
        for key, value in sorted(results.items()):
            writer.writerow([key, value])

    with open(OUT.parent / 'experiment_results.tex', 'w', encoding='utf-8') as file:
        file.write('% Auto-generated by generate_ch03_figures.py\n')
        for key, value in sorted(results.items()):
            command = latex_command_name(key)
            formatted = f'{value:.6g}' if isinstance(value, float) else str(value)
            file.write(f'\\newcommand{{\\{command}}}{{{formatted}}}\n')

    print('Generated', len(results), 'metrics and figures in', OUT, flush=True)
    for key, value in sorted(results.items()):
        print(f'{key}: {value}', flush=True)

if __name__=='__main__':
    main()
