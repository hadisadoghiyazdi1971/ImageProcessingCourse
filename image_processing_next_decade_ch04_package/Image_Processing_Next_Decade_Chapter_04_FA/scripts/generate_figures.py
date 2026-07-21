from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1405)

# Figure 1: different noise models on a smooth image
n = 160
x = np.linspace(-1, 1, n)
X, Y = np.meshgrid(x, x)
clean = 0.15 + 0.75*np.exp(-4*(X**2 + Y**2)) + 0.08*np.sin(10*X)*np.cos(8*Y)
clean = np.clip(clean, 0, 1)

gauss = np.clip(clean + rng.normal(0, 0.08, clean.shape), 0, 1)
peak = 30.0
poisson = rng.poisson(peak*clean)/peak
sp = clean.copy()
mask = rng.random(clean.shape) < 0.08
sp[mask] = rng.integers(0, 2, mask.sum())
speckle = np.clip(clean * rng.gamma(shape=4.0, scale=1/4.0, size=clean.shape), 0, 1)

fig, axes = plt.subplots(1, 5, figsize=(13, 2.8), constrained_layout=True)
for ax, im, title in zip(axes, [clean, gauss, poisson, sp, speckle],
                         ['Clean', 'Gaussian', 'Poisson', 'Impulse', 'Speckle']):
    ax.imshow(im, cmap='gray', vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis('off')
fig.savefig(FIG/'noise_models_examples.pdf', bbox_inches='tight')
fig.savefig(FIG/'noise_models_examples.png', dpi=220, bbox_inches='tight')
plt.close(fig)

# Figure 2: covariance and PSD of white vs correlated fields
m = 512
white = rng.normal(size=(m, m))
# Correlated field by separable smoothing in Fourier domain
fy = np.fft.fftfreq(m)
fx = np.fft.fftfreq(m)
FX, FY = np.meshgrid(fx, fy)
H = np.exp(-0.5*(FX**2 + FY**2)/(0.035**2))
corr = np.fft.ifft2(np.fft.fft2(white)*H).real
corr = (corr - corr.mean())/corr.std()

# empirical horizontal autocorrelation
lags = np.arange(0, 80)
def acf_rows(z):
    z = z - z.mean()
    vals = []
    denom = np.mean(z*z)
    for k in lags:
        vals.append(np.mean(z[:, :m-k]*z[:, k:])/denom)
    return np.array(vals)

acf_w = acf_rows(white)
acf_c = acf_rows(corr)
psd_w = np.fft.fftshift(np.abs(np.fft.fft2(white))**2)
psd_c = np.fft.fftshift(np.abs(np.fft.fft2(corr))**2)
rad_w = np.mean(psd_w, axis=0)
rad_c = np.mean(psd_c, axis=0)
faxis = np.fft.fftshift(np.fft.fftfreq(m))

fig, ax = plt.subplots(figsize=(6.5, 4.0))
ax.plot(lags, acf_w, label='white')
ax.plot(lags, acf_c, label='correlated')
ax.set_xlabel('spatial lag')
ax.set_ylabel('normalized autocovariance')
ax.grid(True, alpha=0.25)
ax.legend()
fig.savefig(FIG/'autocovariance_white_correlated.pdf', bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 4.0))
ax.semilogy(faxis, rad_w/rad_w.max(), label='white')
ax.semilogy(faxis, rad_c/rad_c.max(), label='correlated')
ax.set_xlabel('normalized spatial frequency')
ax.set_ylabel('normalized power')
ax.grid(True, alpha=0.25)
ax.legend()
fig.savefig(FIG/'psd_white_correlated.pdf', bbox_inches='tight')
plt.close(fig)

# Figure 3: Photon transfer curve simulation
levels = np.linspace(5, 1000, 28)
gain = 0.42
read_var = 5.5
means = []
vars_ = []
for level in levels:
    frames = gain*rng.poisson(level/gain, size=(40, 1800)) + rng.normal(0, np.sqrt(read_var), (40, 1800))
    means.append(frames.mean())
    # pair-difference estimator removes fixed pattern component
    d = frames[0::2] - frames[1::2]
    vars_.append(0.5*np.var(d, ddof=1))
means = np.asarray(means)
vars_ = np.asarray(vars_)
A = np.c_[means, np.ones_like(means)]
coef, *_ = np.linalg.lstsq(A, vars_, rcond=None)
fit = A @ coef
np.savetxt(ROOT/'outputs'/'ptc_fit.csv', np.c_[means, vars_, fit], delimiter=',', header='mean,variance,fit', comments='')
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.scatter(means, vars_, s=22, label='simulated measurements')
ax.plot(means, fit, label=f'fit: var={coef[0]:.3f} mean+{coef[1]:.3f}')
ax.set_xlabel('mean digital signal')
ax.set_ylabel('temporal variance')
ax.grid(True, alpha=0.25)
ax.legend()
fig.savefig(FIG/'photon_transfer_curve.pdf', bbox_inches='tight')
plt.close(fig)

# Figure 4: variance stabilization by Anscombe
lam = np.geomspace(0.2, 200, 30)
raw_var = []
ans_var = []
for l in lam:
    y = rng.poisson(l, size=200000)
    raw_var.append(np.var(y))
    z = 2*np.sqrt(y + 3/8)
    ans_var.append(np.var(z))
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.semilogx(lam, raw_var, label='raw Poisson variance')
ax.semilogx(lam, ans_var, label='Anscombe-domain variance')
ax.axhline(1.0, linestyle='--', linewidth=1, label='target variance 1')
ax.set_xlabel('Poisson mean')
ax.set_ylabel('empirical variance')
ax.grid(True, alpha=0.25)
ax.legend()
fig.savefig(FIG/'anscombe_variance.pdf', bbox_inches='tight')
plt.close(fig)

# Figure 5: residual diagnostics for correct and incorrect noise models
truth = np.linspace(0.02, 1.0, 20000)
y = rng.poisson(60*truth)/60 + rng.normal(0, 0.012, truth.shape)
res_raw = y-truth
sigma2 = truth/60 + 0.012**2
res_std = res_raw/np.sqrt(sigma2)
bins = np.linspace(0,1,16)
idx = np.digitize(truth,bins)-1
centers = (bins[:-1]+bins[1:])/2
v_raw = np.array([np.var(res_raw[idx==i]) for i in range(len(centers))])
v_std = np.array([np.var(res_std[idx==i]) for i in range(len(centers))])
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(centers, v_raw, marker='o', label='raw residual variance')
ax.plot(centers, v_std, marker='s', label='standardized residual variance')
ax.set_xlabel('signal level')
ax.set_ylabel('variance within bin')
ax.grid(True, alpha=0.25)
ax.legend()
fig.savefig(FIG/'residual_heteroscedasticity.pdf', bbox_inches='tight')
plt.close(fig)

print('Figures generated in', FIG)
