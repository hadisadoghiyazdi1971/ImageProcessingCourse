from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

def box(ax, xy, w, h, text, fc="#eef4fb", ec="#264d73", fs=9):
    p = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=1.5)
    ax.add_patch(p)
    ax.text(xy[0]+w/2, xy[1]+h/2, text, ha="center", va="center", fontsize=fs)
    return p

def arrow(ax, p1, p2, text=None):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12, lw=1.3, color="#333333")
    ax.add_patch(a)
    if text:
        ax.text((p1[0]+p2[0])/2, (p1[1]+p2[1])/2+0.04, text, ha="center", fontsize=8)

# 1 taxonomy
fig, ax = plt.subplots(figsize=(9.2,4.8)); ax.axis('off'); ax.set_xlim(0,10); ax.set_ylim(0,6)
box(ax,(4,4.8),2,0.7,"Deep generative models",fc="#fff4df")
items=[("Latent-variable\nVAE / VQ",0.4,3.1),("Adversarial\nGAN",2.4,3.1),("Autoregressive\nPixel / Token",4.4,3.1),("Normalizing flow\nExact density",6.4,3.1),("Energy-based\nUnnormalized",8.4,3.1)]
for t,x,y in items:
    box(ax,(x,y),1.3,0.9,t); arrow(ax,(5,4.8),(x+0.65,y+0.9))
for x,txt in [(0.4,"ELBO\nAmortized inference"),(2.4,"Game / IPM\nImplicit density"),(4.4,"Chain rule\nSequential sampling"),(6.4,"Change of variables\nInvertibility"),(8.4,"Energy landscape\nMCMC / score")]:
    box(ax,(x,1.3),1.3,0.9,txt,fc="#eff8ef",ec="#3f7d4a",fs=8); arrow(ax,(x+0.65,3.1),(x+0.65,2.2))
ax.set_title("A unified taxonomy: what is normalized, what is inferred, and how are samples obtained?")
save(fig,"fig12_01_taxonomy")

# 2 pushforward
rng=np.random.default_rng(2)
z=rng.normal(size=(600,2)); x=np.c_[z[:,0],0.35*z[:,1]+0.25*np.sin(2*z[:,0])]; x[:,0]=np.tanh(1.1*x[:,0])
fig,axs=plt.subplots(1,3,figsize=(9.2,3.0))
axs[0].scatter(z[:,0],z[:,1],s=5,alpha=.35); axs[0].set_title("Base latent $p(z)$")
axs[1].axis('off'); axs[1].text(.5,.5,"$x=G_\\theta(z)$\nPushforward",ha='center',va='center',fontsize=14); axs[1].annotate('',xy=(.9,.5),xytext=(.1,.5),arrowprops=dict(arrowstyle='->',lw=2))
axs[2].scatter(x[:,0],x[:,1],s=5,alpha=.35); axs[2].set_title("Model samples $p_\\theta(x)$")
for a in (axs[0],axs[2]): a.set_aspect('equal'); a.grid(alpha=.2)
save(fig,"fig12_02_pushforward")

# 3 divergence intuition
xx=np.linspace(-5,5,1200); p=np.exp(-.5*((xx+1.3)/.65)**2); p/=np.trapezoid(p,xx); q=np.exp(-.5*((xx-1.2)/.8)**2); q/=np.trapezoid(q,xx)
fig,axs=plt.subplots(1,2,figsize=(9.2,3.2))
axs[0].plot(xx,p,label='data p'); axs[0].plot(xx,q,label='model q'); axs[0].fill_between(xx,np.minimum(p,q),alpha=.25); axs[0].legend(); axs[0].set_title("Disjoint support: KL/JS gradients may saturate")
for mu in np.linspace(-2.5,2.5,60):
    qq=np.exp(-.5*((xx-mu)/.8)**2); qq/=np.trapezoid(qq,xx)
    kl=np.trapezoid(p*np.log((p+1e-12)/(qq+1e-12)),xx)
    w=abs(mu+1.3)
    axs[1].scatter(mu,kl,s=6)
axs[1].plot(np.linspace(-2.5,2.5,60),abs(np.linspace(-2.5,2.5,60)+1.3),lw=2,label='Wasserstein proxy')
axs[1].set_yscale('log'); axs[1].set_title("Objective geometry"); axs[1].legend(); axs[1].grid(alpha=.2)
save(fig,"fig12_03_divergences")

# 4 AE VAE
fig, ax=plt.subplots(figsize=(9.2,3.8)); ax.axis('off'); ax.set_xlim(0,12); ax.set_ylim(0,4)
box(ax,(.3,1.4),1.5,1,"image x"); box(ax,(2.5,1.4),1.7,1,"encoder"); box(ax,(5,2.2),1.6,.8,"$\\mu_\\phi(x)$"); box(ax,(5,.8),1.6,.8,"$\\log \\sigma^2_\\phi(x)$")
box(ax,(7.2,1.4),1.7,1,"$z=\\mu+\\sigma\\odot\\epsilon$",fc="#fff4df"); box(ax,(9.6,1.4),1.6,1,"decoder"); box(ax,(11.5,1.4),.4,1,"$\\hat x$")
for p1,p2 in [((1.8,1.9),(2.5,1.9)),((4.2,1.9),(5,2.6)),((4.2,1.9),(5,1.2)),((6.6,2.6),(7.2,1.9)),((6.6,1.2),(7.2,1.9)),((8.9,1.9),(9.6,1.9)),((11.2,1.9),(11.5,1.9))]: arrow(ax,p1,p2)
ax.text(7.9,.55,"$\\epsilon\\sim\\mathcal{N}(0,I)$",ha='center'); ax.set_title("VAE computational graph and the reparameterization path")
save(fig,"fig12_04_vae_graph")

# 5 ELBO
betas=np.linspace(.05,4,100); rate=2.6/(betas+.2)+.1; distortion=.45+.55*np.log1p(betas)
fig,axs=plt.subplots(1,2,figsize=(9.2,3.2))
axs[0].bar([0,1],[2.3,.7]); axs[0].set_xticks([0,1],["log p(x)","variational gap"]); axs[0].set_title("$\\log p(x)=ELBO+KL(q||p)$")
axs[1].plot(rate,distortion); axs[1].scatter(rate[::15],distortion[::15],s=20); axs[1].set_xlabel("rate  KL(q(z|x)||p(z))"); axs[1].set_ylabel("distortion"); axs[1].set_title("Rate-distortion frontier controlled by $\\beta$"); axs[1].grid(alpha=.2)
save(fig,"fig12_05_elbo_rate_distortion")

# 6 reparam gradients
fig,axs=plt.subplots(1,2,figsize=(9.2,3.2)); n=5000; eps=rng.normal(size=n)
mu=.7; sigma=.8; direct=((mu+sigma*eps)**2)
axs[0].hist(direct,bins=45,density=True,alpha=.7); axs[0].set_title("Samples after $z=\\mu+\\sigma\\epsilon$")
for m in [-1,-.5,0,.5,1]:
    g=2*(m+sigma*eps)
    axs[1].scatter(m,np.mean(g),s=30)
axs[1].plot([-1,1],[-2,2]); axs[1].set_title("Pathwise gradient: low-variance derivative"); axs[1].grid(alpha=.2)
save(fig,"fig12_06_reparameterization")

# 7 posterior collapse
steps=np.arange(250); kl_collapse=1.6*np.exp(-steps/25); kl_free=.35+1.1*np.exp(-steps/60); recon=1.5-.8*(1-np.exp(-steps/55))
fig,ax=plt.subplots(figsize=(8.8,3.2)); ax.plot(steps,kl_collapse,label='KL collapse'); ax.plot(steps,kl_free,label='KL with free-bits/anneal'); ax.plot(steps,recon,label='reconstruction'); ax.legend(); ax.set_xlabel('training step'); ax.set_title('Posterior collapse diagnostics'); ax.grid(alpha=.2); save(fig,"fig12_07_posterior_collapse")

# 8 VQ
centers=np.array([[-1,-1],[-1,1],[1,-1],[1,1],[0,0]]); pts=np.vstack([c+.28*rng.normal(size=(100,2)) for c in centers]); d=((pts[:,None,:]-centers[None,:,:])**2).sum(-1); lab=d.argmin(1)
fig,axs=plt.subplots(1,2,figsize=(9.2,3.4)); axs[0].scatter(pts[:,0],pts[:,1],c=lab,s=8); axs[0].scatter(centers[:,0],centers[:,1],marker='x',s=100,c='black'); axs[0].set_title('Nearest-code vector quantization')
counts=np.bincount(lab,minlength=len(centers)); probs=counts/counts.sum(); perplex=np.exp(-(probs*np.log(probs+1e-12)).sum()); axs[1].bar(np.arange(len(counts)),counts); axs[1].set_title(f'Code usage and perplexity = {perplex:.2f}'); save(fig,"fig12_08_vq_codebook")

# 9 VAE variants map
fig,ax=plt.subplots(figsize=(9.2,4.4)); ax.axis('off'); ax.set_xlim(0,10); ax.set_ylim(0,5)
box(ax,(4,4),2,0.65,"VAE family",fc="#fff4df")
variants=[("$\\beta$-VAE\ncapacity",.4,2.5),("IWAE\ntighter bound",2.3,2.5),("WAE/InfoVAE\naggregate match",4.2,2.5),("Hierarchical VAE\nmultiscale latents",6.1,2.5),("VQ-VAE/VQGAN\ndiscrete latents",8.0,2.5)]
for t,x,y in variants: box(ax,(x,y),1.5,.85,t); arrow(ax,(5,4),(x+.75,y+.85))
box(ax,(3.7,.75),2.6,.85,"Common failure: posterior/codebook collapse",fc="#fcecec",ec="#9c3333"); arrow(ax,(5,2.5),(5,1.6))
ax.set_title('VAE extensions modify the bound, prior, posterior, hierarchy, or latent alphabet'); save(fig,"fig12_09_vae_family")

# 10 GAN game
fig,ax=plt.subplots(figsize=(9.2,3.8)); ax.axis('off'); ax.set_xlim(0,11); ax.set_ylim(0,4)
box(ax,(.3,1.5),1.5,.9,"$z\\sim p(z)$"); box(ax,(2.4,1.5),1.6,.9,"Generator G"); box(ax,(4.8,2.2),1.5,.9,"fake $G(z)$"); box(ax,(4.8,.8),1.5,.9,"real $x$"); box(ax,(7.1,1.5),1.7,.9,"Discriminator D"); box(ax,(9.7,1.5),.9,.9,"score")
for p1,p2 in [((1.8,1.95),(2.4,1.95)),((4,1.95),(4.8,2.65)),((6.3,2.65),(7.1,1.95)),((6.3,1.25),(7.1,1.95)),((8.8,1.95),(9.7,1.95))]: arrow(ax,p1,p2)
ax.annotate('adversarial gradient',xy=(3.2,1.45),xytext=(7.9,.35),arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.25')); ax.set_title('A GAN is a coupled optimization game, not a single minimization problem'); save(fig,"fig12_10_gan_game")

# 11 optimal discriminator
xx=np.linspace(-4,4,600); p=np.exp(-.5*((xx+1)/.9)**2); q=np.exp(-.5*((xx-1)/1.1)**2); d=p/(p+q+1e-12)
fig,ax=plt.subplots(figsize=(8.8,3.2)); ax.plot(xx,p/p.max(),label='scaled data density'); ax.plot(xx,q/q.max(),label='scaled model density'); ax.plot(xx,d,label='$D^*(x)=p/(p+q)$'); ax.legend(); ax.set_title('Optimal discriminator and density-ratio estimation'); ax.grid(alpha=.2); save(fig,"fig12_11_optimal_discriminator")

# 12 mode collapse
fig,axs=plt.subplots(1,3,figsize=(9.2,3.0)); modes=np.array([[np.cos(t),np.sin(t)] for t in np.linspace(0,2*np.pi,8,endpoint=False)])
real=np.vstack([m+.08*rng.normal(size=(60,2)) for m in modes]); collapsed=np.vstack([m+.08*rng.normal(size=(90,2)) for m in modes[:3]]); covered=np.vstack([m+.08*rng.normal(size=(60,2)) for m in modes])
for a,data,title in zip(axs,[real,collapsed,covered],["data modes","mode collapse","improved coverage"]): a.scatter(data[:,0],data[:,1],s=7); a.set_aspect('equal'); a.set_title(title); a.axis('off')
save(fig,"fig12_12_mode_collapse")

# 13 WGAN critic
x=np.linspace(-3,3,500); critic=np.tanh(1.2*x); grad=np.gradient(critic,x)
fig,axs=plt.subplots(1,2,figsize=(9.2,3.2)); axs[0].plot(x,critic); axs[0].set_title('1-Lipschitz critic potential'); axs[1].plot(x,abs(grad)); axs[1].axhline(1,ls='--'); axs[1].set_title('Gradient-norm regularization'); [a.grid(alpha=.2) for a in axs]; save(fig,"fig12_13_wgan_lipschitz")

# 14 training dynamics
steps=np.arange(500); dloss=.7+.18*np.sin(steps/13)*np.exp(-steps/500); gloss=1.1+.25*np.sin(steps/13+1.4)*np.exp(-steps/500); fid=65*np.exp(-steps/140)+8+2*np.sin(steps/35)
fig,axs=plt.subplots(1,2,figsize=(9.2,3.2)); axs[0].plot(steps,dloss,label='D loss'); axs[0].plot(steps,gloss,label='G loss'); axs[0].legend(); axs[0].set_title('Losses can oscillate without indicating failure'); axs[1].plot(steps,fid); axs[1].set_title('Use sample/coverage metrics and diagnostics'); [a.grid(alpha=.2) for a in axs]; save(fig,"fig12_14_gan_dynamics")

# 15 StyleGAN
fig,ax=plt.subplots(figsize=(9.2,4)); ax.axis('off'); ax.set_xlim(0,12); ax.set_ylim(0,4)
box(ax,(.2,1.55),1.2,.9,"z"); box(ax,(2,1.55),1.6,.9,"mapping\nnetwork"); box(ax,(4.3,1.55),1.3,.9,"w / w+");
for i,x0 in enumerate([6.2,7.8,9.4]): box(ax,(x0,1.4),1.15,1.2,f"synthesis\nblock {i+1}",fc="#eff8ef",ec="#3f7d4a")
box(ax,(11.1,1.55),.7,.9,"image")
for p1,p2 in [((1.4,2),(2,2)),((3.6,2),(4.3,2)),((5.6,2),(6.2,2)),((7.35,2),(7.8,2)),((8.95,2),(9.4,2)),((10.55,2),(11.1,2))]: arrow(ax,p1,p2)
ax.text(8.35,.55,"style modulation + noise + controlled resampling",ha='center'); ax.set_title('Style-based synthesis separates latent mapping from spatial synthesis'); save(fig,"fig12_15_stylegan")

# 16 AR chain
fig,ax=plt.subplots(figsize=(9.2,3.5)); ax.axis('off'); ax.set_xlim(0,12); ax.set_ylim(0,3)
for i in range(8):
    box(ax,(.3+i*1.4,1.25),.95,.75,f"$x_{i+1}$",fc="#eef4fb" if i<5 else "#fff4df")
    if i<7: arrow(ax,(1.25+i*1.4,1.62),(1.7+i*1.4,1.62))
ax.text(6,2.5,"$p(x)=\\prod_i p(x_i\\mid x_{<i})$",ha='center',fontsize=14); ax.set_title('Autoregressive factorization: exact likelihood, sequential sampling'); save(fig,"fig12_16_autoregressive_chain")

# 17 masked conv
mask=np.tril(np.ones((7,7))); mask[3:,4:]=0
fig,axs=plt.subplots(1,2,figsize=(8.8,3.5)); axs[0].imshow(mask,cmap='gray_r'); axs[0].set_title('Causal receptive mask'); axs[0].axis('off')
grid=np.arange(64).reshape(8,8); axs[1].imshow(grid,cmap='viridis'); axs[1].plot([-.5,7.5],[3.5,3.5],color='white'); axs[1].set_title('Raster order imposes an artificial sequence'); axs[1].axis('off'); save(fig,"fig12_17_masked_convolution")

# 18 tokenizer+AR
fig,ax=plt.subplots(figsize=(9.2,3.8)); ax.axis('off'); ax.set_xlim(0,12); ax.set_ylim(0,4)
box(ax,(.3,1.5),1.4,.9,"image"); box(ax,(2.2,1.5),1.5,.9,"tokenizer"); box(ax,(4.3,1.5),1.8,.9,"discrete tokens"); box(ax,(6.8,1.5),1.8,.9,"AR / VAR model"); box(ax,(9.3,1.5),1.4,.9,"decoder"); box(ax,(11.2,1.5),.6,.9,"image")
for x1,x2 in [(1.7,2.2),(3.7,4.3),(6.1,6.8),(8.6,9.3),(10.7,11.2)]: arrow(ax,(x1,1.95),(x2,1.95))
ax.text(5.2,.55,"Tokenizer quality sets the upper bound; sequence design sets generation cost.",ha='center'); ax.set_title('Two-stage visual generation with learned discrete tokens'); save(fig,"fig12_18_token_ar")

# 19 flow coupling
fig,ax=plt.subplots(figsize=(9.2,4)); ax.axis('off'); ax.set_xlim(0,11); ax.set_ylim(0,4)
box(ax,(.4,2.2),1.2,.75,"$x_a$"); box(ax,(.4,.9),1.2,.75,"$x_b$"); box(ax,(3,2.2),1.4,.75,"$y_a=x_a$"); box(ax,(3,.9),2.1,.75,"$y_b=x_b\\odot e^{s(x_a)}+t(x_a)$",fs=8); box(ax,(7.1,1.55),2.2,.9,"triangular Jacobian"); box(ax,(9.8,1.55),.9,.9,"exact log-det")
for p1,p2 in [((1.6,2.57),(3,2.57)),((1.6,1.27),(3,1.27)),((4.4,2.57),(7.1,2)),((5.1,1.27),(7.1,2)),((9.3,2),(9.8,2))]: arrow(ax,p1,p2)
ax.set_title('Affine coupling trades unrestricted transforms for tractable Jacobians'); save(fig,"fig12_19_flow_coupling")

# 20 flow density
z=rng.normal(size=(500,2)); t=np.linspace(0,1,5)
fig,axs=plt.subplots(1,5,figsize=(10,2.2))
for a,tt in zip(axs,t):
    y=np.c_[z[:,0]+tt*.8*np.sin(z[:,1]*1.7), z[:,1]+tt*.55*np.sin(z[:,0]*1.4)]
    a.scatter(y[:,0],y[:,1],s=4,alpha=.35); a.set_title(f't={tt:.2f}'); a.axis('off'); a.set_aspect('equal')
fig.suptitle('Continuous normalizing flow / flow matching transports a base distribution through time')
save(fig,"fig12_20_continuous_flow")

# 21 EBM landscape
xx=np.linspace(-3,3,400); E=.25*(xx**2-2)**2+.12*xx; p=np.exp(-E); p/=np.trapezoid(p,xx)
fig,axs=plt.subplots(1,2,figsize=(9.2,3.2)); axs[0].plot(xx,E); axs[0].set_title('Energy $E_\\theta(x)$'); axs[1].plot(xx,p); axs[1].fill_between(xx,p,alpha=.25); axs[1].set_title('$p_\\theta(x)\\propto e^{-E_\\theta(x)}$'); [a.grid(alpha=.2) for a in axs]; save(fig,"fig12_21_energy_model")

# 22 metrics PR
fig,axs=plt.subplots(1,2,figsize=(9.2,3.4));
real=np.vstack([rng.normal([-1,0],[.5,.35],(400,2)),rng.normal([1,0],[.45,.45],(400,2))]); hi_prec=rng.normal([-1,0],[.38,.27],(400,2)); hi_rec=np.vstack([rng.normal([-1,0],[.7,.55],(250,2)),rng.normal([1,0],[.7,.55],(250,2))])
axs[0].scatter(real[:,0],real[:,1],s=4,alpha=.12,label='real'); axs[0].scatter(hi_prec[:,0],hi_prec[:,1],s=6,alpha=.35,label='high precision, low recall'); axs[0].legend(fontsize=7); axs[0].set_title('Quality without coverage')
axs[1].scatter(real[:,0],real[:,1],s=4,alpha=.12,label='real'); axs[1].scatter(hi_rec[:,0],hi_rec[:,1],s=6,alpha=.3,label='higher recall, lower precision'); axs[1].legend(fontsize=7); axs[1].set_title('Coverage with off-manifold samples')
for a in axs: a.set_aspect('equal'); a.axis('off')
save(fig,"fig12_22_precision_recall")

# 23 library map
fig,ax=plt.subplots(figsize=(9.2,5)); ax.axis('off'); ax.set_xlim(0,10); ax.set_ylim(0,6)
levels=[("Research models",4.0,["StyleGAN3","taming-transformers","VAR/LlamaGen"]),("Generative toolboxes",3.0,["MMagic","Diffusers VAE/VQ","MONAI Generative"]),("Metrics and tracking",2.0,["torch-fidelity","Clean-FID","W&B / MLflow"]),("Core",1.0,["PyTorch","torch.distributions","torchvision"])]
for title,y,names in levels:
    ax.text(.25,y+.25,title,ha='left',va='center',weight='bold')
    for j,n in enumerate(names): box(ax,(2.3+j*2.35,y),1.9,.55,n,fc="#eef4fb" if y<3 else "#eff8ef",fs=8)
ax.set_title('Library stack: mathematics first, framework second, pretrained systems last'); save(fig,"fig12_23_libraries")

# 24 evaluation protocol
fig,ax=plt.subplots(figsize=(9.2,4.4)); ax.axis('off'); ax.set_xlim(0,12); ax.set_ylim(0,5)
steps=[("split / deduplicate",.2),("train + log",2.25),("sample fixed N",4.3),("quality + coverage",6.35),("memorization / bias",8.4),("human + task audit",10.45)]
for i,(t,x0) in enumerate(steps):
    box(ax,(x0,2),1.35,.9,t,fc="#eef4fb" if i<3 else "#eff8ef",fs=8)
    if i<len(steps)-1: arrow(ax,(x0+1.35,2.45),(steps[i+1][1],2.45))
ax.text(6,3.8,"No single number is a sufficient evaluation of a generative model",ha='center',fontsize=13)
ax.text(6,.8,"Report likelihood/ELBO when available, FID/KID, precision-recall, nearest-neighbor audits, conditional fidelity, cost, and failure cases.",ha='center',fontsize=8)
save(fig,"fig12_24_evaluation")

# 25 timeline/frontier
fig,ax=plt.subplots(figsize=(9.2,3.8)); ax.set_xlim(2013,2027); ax.set_ylim(0,5); ax.set_yticks([]); ax.grid(axis='x',alpha=.2)
events=[(2013,1,'VAE'),(2014,2,'GAN'),(2016,1.2,'PixelCNN / RealNVP'),(2017,2.2,'WGAN-GP / VQ-VAE'),(2018,3,'Glow / BigGAN'),(2019,2,'StyleGAN'),(2020,3.3,'StyleGAN2 / VQGAN'),(2021,2.3,'StyleGAN3'),(2023,3.4,'Flow Matching'),(2024,2.1,'VAR / LlamaGen / TiTok'),(2025,3.1,'STARFlow / OmniGen-AR'),(2026,1.3,'adaptive & semantic tokenizers')]
for year,y,label in events: ax.scatter(year,y,s=45); ax.text(year,y+.22,label,rotation=25,ha='left',fontsize=8)
ax.set_title('The field is converging toward compact latents, scalable sequence models, and continuous transport'); ax.set_xlabel('year'); save(fig,"fig12_25_frontier")

(OUT.parent/"outputs"/"figure_manifest.txt").write_text("\n".join(sorted(p.name for p in OUT.iterdir())),encoding='utf-8')
print(f"generated {len(list(OUT.glob('*.pdf')))} PDF figures")
