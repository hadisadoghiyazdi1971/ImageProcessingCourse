from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, convolve

OUT = Path(__file__).resolve().parent / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1e-12


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=220, bbox_inches='tight')
    plt.close(fig)


def zone_plate(n=768):
    u = np.linspace(-1, 1, n, endpoint=False)
    x, y = np.meshgrid(u, u)
    r2 = x*x + y*y
    z = 0.5 + 0.5*np.cos(190*np.pi*r2)
    z *= np.exp(-0.25*r2)
    return np.clip(z, 0, 1)


def fft_log(img):
    f = np.fft.fftshift(np.fft.fft2(img))
    return np.log1p(np.abs(f))


def fig_sampling():
    z = zone_plate()
    factor = 6
    naive = z[::factor, ::factor]
    pre = gaussian_filter(z, sigma=2.4)[::factor, ::factor]
    fig, ax = plt.subplots(2, 3, figsize=(12, 7.4))
    ax[0,0].imshow(z, cmap='gray', vmin=0, vmax=1)
    ax[0,0].set_title('Continuous-like test pattern')
    ax[0,1].imshow(naive, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax[0,1].set_title('Naive decimation')
    ax[0,2].imshow(pre, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax[0,2].set_title('Prefilter + decimation')
    ax[1,0].imshow(fft_log(z), cmap='magma')
    ax[1,0].set_title('Original log spectrum')
    ax[1,1].imshow(fft_log(naive), cmap='magma')
    ax[1,1].set_title('Aliased log spectrum')
    ax[1,2].imshow(fft_log(pre), cmap='magma')
    ax[1,2].set_title('Controlled spectrum')
    for a in ax.ravel(): a.axis('off')
    savefig(fig, 'sampling_aliasing_zoneplate.png')


def fig_quantization():
    rng = np.random.default_rng(1405)
    n = 768
    x = np.linspace(0,1,n)
    grad = np.tile(x, (180,1))
    b = 3
    levels = 2**b
    delta = 1/(levels-1)
    q = np.round(grad/delta)*delta
    dither = rng.uniform(-delta/2, delta/2, size=grad.shape)
    qd = np.clip(np.round((grad+dither)/delta)*delta,0,1)
    err = q-grad
    errd = qd-grad
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.7))
    ax[0,0].imshow(grad, cmap='gray', aspect='auto', vmin=0, vmax=1)
    ax[0,0].set_title('Ideal linear ramp')
    ax[0,1].imshow(q, cmap='gray', aspect='auto', vmin=0, vmax=1)
    ax[0,1].set_title('3-bit quantization')
    ax[0,2].imshow(qd, cmap='gray', aspect='auto', vmin=0, vmax=1)
    ax[0,2].set_title('3-bit + uniform dither')
    ax[1,0].plot(x, grad[0], label='ideal')
    ax[1,0].plot(x, q[0], label='quantized', linewidth=1)
    ax[1,0].set_title('Transfer along one row')
    ax[1,0].legend()
    ax[1,1].hist(err.ravel(), bins=80, density=True)
    ax[1,1].set_title(f'Error, MSE={np.mean(err**2):.4e}')
    ax[1,2].hist(errd.ravel(), bins=80, density=True)
    ax[1,2].set_title(f'Dithered error, MSE={np.mean(errd**2):.4e}')
    for a in ax[0]: a.axis('off')
    for a in ax[1]: a.grid(alpha=0.25)
    savefig(fig, 'quantization_and_dither.png')


def fig_sensor_snr():
    photons = np.logspace(-2, 6, 800)
    fig, ax = plt.subplots(figsize=(8.5,5.4))
    for sigma_r in [0.5, 2.0, 8.0, 32.0]:
        snr = photons / np.sqrt(photons + sigma_r**2)
        ax.semilogx(photons, 20*np.log10(np.maximum(snr, EPS)), label=f'read noise={sigma_r:g} e-')
    ax.semilogx(photons, 10*np.log10(np.maximum(photons, EPS)), '--', label='shot-noise limit')
    ax.set_xlabel('Mean detected photoelectrons')
    ax.set_ylabel('SNR (dB)')
    ax.set_title('Shot noise, read noise, and the two SNR regimes')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    savefig(fig, 'sensor_snr_regimes.png')


def synthetic_rgb(n=384):
    y, x = np.mgrid[0:n,0:n]
    xn, yn = x/(n-1), y/(n-1)
    img = np.stack([
        0.15 + 0.75*xn,
        0.1 + 0.8*yn,
        0.15 + 0.65*(1-xn*yn)
    ], axis=-1)
    circle = (xn-0.32)**2 + (yn-0.35)**2 < 0.12**2
    img[circle] = np.array([0.95,0.15,0.1])
    rect = (xn>0.58)&(xn<0.86)&(yn>0.18)&(yn<0.43)
    img[rect] = np.array([0.08,0.85,0.22])
    stripes = (np.sin(2*np.pi*(18*xn+6*yn))>0) & (yn>0.62)
    img[stripes] = 0.88*img[stripes]
    img[~stripes & (yn>0.62)] *= 0.42
    return np.clip(img,0,1)


def bayer_mosaic(rgb):
    h,w,_ = rgb.shape
    raw = np.zeros((h,w))
    masks=[]
    R=np.zeros((h,w),bool); G=np.zeros((h,w),bool); B=np.zeros((h,w),bool)
    R[0::2,0::2]=True
    G[0::2,1::2]=True; G[1::2,0::2]=True
    B[1::2,1::2]=True
    raw[R]=rgb[...,0][R]; raw[G]=rgb[...,1][G]; raw[B]=rgb[...,2][B]
    return raw,(R,G,B)


def bilinear_demosaic(raw,masks):
    R,G,B=masks
    kernels=[np.array([[1,2,1],[2,4,2],[1,2,1]],float)]*3
    out=[]
    for M,K in zip([R,G,B],kernels):
        num=convolve(raw*M,K,mode='mirror')
        den=convolve(M.astype(float),K,mode='mirror')
        out.append(num/np.maximum(den,EPS))
    return np.clip(np.stack(out,-1),0,1)


def fig_bayer():
    rgb=synthetic_rgb()
    raw,masks=bayer_mosaic(rgb)
    rec=bilinear_demosaic(raw,masks)
    err=np.abs(rec-rgb)
    psnr=10*np.log10(1/np.mean((rec-rgb)**2))
    vis=np.zeros_like(rgb)
    R,G,B=masks
    vis[...,0][R]=raw[R]; vis[...,1][G]=raw[G]; vis[...,2][B]=raw[B]
    fig,ax=plt.subplots(1,4,figsize=(14,4.1))
    ax[0].imshow(rgb); ax[0].set_title('Reference linear RGB')
    ax[1].imshow(vis); ax[1].set_title('RGGB samples')
    ax[2].imshow(rec); ax[2].set_title(f'Bilinear demosaic\nPSNR={psnr:.2f} dB')
    ax[3].imshow(np.clip(5*err,0,1)); ax[3].set_title('5x absolute error')
    for a in ax:a.axis('off')
    savefig(fig,'bayer_demosaicing.png')


def srgb_encode(L):
    return np.where(L<=0.0031308,12.92*L,1.055*np.power(L,1/2.4)-0.055)


def pq_encode(L):
    # L normalized to 10000 cd/m^2
    m1=2610/16384; m2=2523/4096*128
    c1=3424/4096; c2=2413/4096*32; c3=2392/4096*32
    p=np.power(np.clip(L,0,1),m1)
    return np.power((c1+c2*p)/(1+c3*p),m2)


def hlg_encode(E):
    a=0.17883277; b=1-4*a; c=0.5-a*np.log(4*a)
    return np.where(E<=1/12,np.sqrt(3*E),a*np.log(12*E-b)+c)


def fig_transfer():
    L=np.linspace(0,1,2001)
    fig,ax=plt.subplots(1,2,figsize=(11,4.4))
    ax[0].plot(L,L,label='Linear')
    ax[0].plot(L,srgb_encode(L),label='sRGB OETF')
    ax[0].plot(L,hlg_encode(np.maximum(L,1e-12)),label='HLG OETF')
    ax[0].plot(L,pq_encode(L),label='PQ inverse EOTF')
    ax[0].set_xlabel('Normalized linear scene/display signal')
    ax[0].set_ylabel('Encoded signal')
    ax[0].set_title('Transfer functions')
    ax[0].grid(alpha=.3); ax[0].legend()
    ax[1].semilogx(np.maximum(L,1e-6),L,label='Linear')
    ax[1].semilogx(np.maximum(L,1e-6),srgb_encode(L),label='sRGB')
    ax[1].semilogx(np.maximum(L,1e-6),hlg_encode(np.maximum(L,1e-12)),label='HLG')
    ax[1].semilogx(np.maximum(L,1e-6),pq_encode(L),label='PQ')
    ax[1].set_xlabel('Input (log axis)')
    ax[1].set_ylabel('Encoded signal')
    ax[1].set_title('Allocation of code values near black')
    ax[1].grid(alpha=.3); ax[1].legend()
    savefig(fig,'transfer_functions_srgb_pq_hlg.png')


def gaussian(lam,mu,sig):
    return np.exp(-0.5*((lam-mu)/sig)**2)


def fig_spectral_metamer():
    lam=np.linspace(400,700,61)
    S=np.vstack([gaussian(lam,610,45),gaussian(lam,545,38),gaussian(lam,455,30)])
    S/=S.max(axis=1,keepdims=True)
    base=0.42+0.18*np.sin((lam-400)/300*2*np.pi)+0.12*np.cos((lam-400)/300*5*np.pi)
    base=np.clip(base,0.12,0.82)
    # a null-space perturbation of the RGB sensing matrix
    u,s,vh=np.linalg.svd(S,full_matrices=True)
    null=vh[3:]
    v=null[7]
    v/=np.max(np.abs(v))
    alpha=0.23
    sp1=np.clip(base+alpha*v,0,1)
    sp2=np.clip(base-alpha*v,0,1)
    # compensate tiny clipping-induced mismatch through least-squares projection
    target=0.5*(S@sp1+S@sp2)
    for sp in [sp1,sp2]:
        corr=np.linalg.lstsq(S@S.T,target-S@sp,rcond=None)[0]
        sp += S.T@corr
        np.clip(sp,0,1,out=sp)
    rgb1=S@sp1; rgb2=S@sp2
    fig,ax=plt.subplots(1,2,figsize=(11.5,4.6))
    ax[0].plot(lam,S[0],label='R sensitivity')
    ax[0].plot(lam,S[1],label='G sensitivity')
    ax[0].plot(lam,S[2],label='B sensitivity')
    ax[0].set_title('Synthetic spectral sensitivities')
    ax[0].set_xlabel('Wavelength (nm)'); ax[0].grid(alpha=.3); ax[0].legend()
    ax[1].plot(lam,sp1,label='Spectrum 1')
    ax[1].plot(lam,sp2,label='Spectrum 2')
    ax[1].set_title('Different spectra with nearly equal RGB')
    ax[1].set_xlabel('Wavelength (nm)'); ax[1].grid(alpha=.3); ax[1].legend()
    fig.suptitle(f'||RGB1-RGB2||2 = {np.linalg.norm(rgb1-rgb2):.2e}')
    savefig(fig,'spectral_metamerism.png')


if __name__=='__main__':
    fig_sampling()
    fig_quantization()
    fig_sensor_snr()
    fig_bayer()
    fig_transfer()
    fig_spectral_metamer()
    print('Generated:', *sorted(p.name for p in OUT.glob('*.png')), sep='\n- ')
