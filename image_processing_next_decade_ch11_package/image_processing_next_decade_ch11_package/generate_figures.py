from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'
OUT = ROOT / 'outputs'
FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(1405)


def save(fig, name: str):
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(FIG / f'{name}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

# 1. Patchification and token sequence
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.set_axis_off()
H, W, p = 8, 12, 2
x0, y0 = 0.2, 0.4
for i in range(H):
    for j in range(W):
        val = 0.2 + 0.7*(i/H)*0.5 + 0.7*(j/W)*0.5
        ax.add_patch(Rectangle((x0+j*0.18, y0+(H-1-i)*0.18), 0.18, 0.18,
                               facecolor=str(1-val), edgecolor='white', lw=0.4))
for i in range(0, H+1, p):
    ax.plot([x0, x0+W*0.18], [y0+i*0.18, y0+i*0.18], color='tab:blue', lw=1.6)
for j in range(0, W+1, p):
    ax.plot([x0+j*0.18, x0+j*0.18], [y0, y0+H*0.18], color='tab:blue', lw=1.6)
ax.text(x0+W*0.09, y0+H*0.18+0.18, 'image grid', ha='center', fontsize=12)
ax.add_patch(FancyArrowPatch((2.65,1.12),(3.35,1.12),arrowstyle='->',mutation_scale=16,lw=1.5))
for k in range(12):
    ax.add_patch(Rectangle((3.45+k*0.52, 0.72), 0.42, 0.8, facecolor=plt.cm.viridis(k/12), edgecolor='black', lw=0.5))
    ax.text(3.66+k*0.52, 0.6, f'{k+1}', ha='center', fontsize=7)
ax.text(6.35, 1.75, 'patch tokens', ha='center', fontsize=12)
ax.add_patch(FancyArrowPatch((9.65,1.12),(10.25,1.12),arrowstyle='->',mutation_scale=16,lw=1.5))
for k, lab in enumerate(['CLS','z1','z2','...','zN']):
    ax.add_patch(Rectangle((10.35+k*0.65,0.75),0.52,0.74,facecolor='white',edgecolor='tab:red',lw=1.2))
    ax.text(10.61+k*0.65,1.12,lab,ha='center',va='center',fontsize=9)
ax.set_xlim(0,14); ax.set_ylim(0,2.35)
save(fig,'fig11_01_patchify')

# 2. Token count and attention cost
fig, ax = plt.subplots(figsize=(7.5,4.5))
ps = np.array([4, 8, 14, 16, 24, 32])
N = (224//ps)**2
cost = N**2
ax.plot(ps, N, marker='o', label='token count N')
ax.set_xlabel('patch size p (pixels)')
ax.set_ylabel('token count N')
ax2 = ax.twinx()
ax2.plot(ps, cost/cost.max(), marker='s', linestyle='--', label='normalized N^2')
ax2.set_ylabel('normalized quadratic attention cost')
ax.grid(alpha=.3)
lines = ax.get_lines()+ax2.get_lines(); labels=[l.get_label() for l in lines]
ax.legend(lines,labels,loc='upper right')
save(fig,'fig11_02_patch_cost')

# 3. Patch size aliasing schematic
fig, axs = plt.subplots(1,3,figsize=(10,3.2))
x=np.linspace(0,1,256); y=0.5+0.45*np.sin(2*np.pi*18*x)
for ax,pix,title in zip(axs,[4,16,32],['fine patches','medium patches','coarse patches']):
    ax.plot(x,y,lw=1)
    bins=np.linspace(0,1,pix+1)
    vals=[y[(x>=bins[i])&(x<bins[i+1])].mean() for i in range(pix)]
    ax.step(bins[:-1],vals,where='post',lw=2)
    ax.set_title(title); ax.set_ylim(0,1); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('patch averaging can suppress or alias fine structure')
save(fig,'fig11_03_patch_aliasing')

# 4. QKV attention geometry
fig, ax=plt.subplots(figsize=(10,4.3)); ax.set_axis_off()
positions=[1.1,4.0,6.9]
for x0,label in zip(positions,['Q','K','V']):
    ax.add_patch(Rectangle((x0,1.45),1.4,1.0,facecolor='white',edgecolor='black',lw=1.2))
    ax.text(x0+.7,1.95,label,ha='center',va='center',fontsize=18)
ax.add_patch(FancyArrowPatch((2.5,1.95),(3.8,1.95),arrowstyle='<->',mutation_scale=16,lw=1.2))
ax.text(3.15,2.25,'similarity',ha='center',fontsize=10)
ax.add_patch(Rectangle((4.25,.3),2.2,.75,facecolor='white',edgecolor='tab:blue',lw=1.2))
ax.text(5.35,.67,'softmax(QK^T/sqrt(d))',ha='center',fontsize=10)
ax.add_patch(FancyArrowPatch((4.7,1.4),(5.1,1.08),arrowstyle='->',mutation_scale=14))
ax.add_patch(FancyArrowPatch((7.0,1.45),(6.2,1.05),arrowstyle='->',mutation_scale=14))
ax.add_patch(Rectangle((7.85,.5),1.55,.85,facecolor='white',edgecolor='tab:red',lw=1.2))
ax.text(8.62,.92,'weighted sum',ha='center',fontsize=10)
ax.add_patch(FancyArrowPatch((6.45,.68),(7.8,.87),arrowstyle='->',mutation_scale=14))
ax.set_xlim(.5,10); ax.set_ylim(0,3.2)
save(fig,'fig11_04_qkv')

# 5. Multi-head view
fig, ax=plt.subplots(figsize=(10,4)); ax.set_axis_off()
for h in range(4):
    y0=.45+h*.72
    ax.add_patch(Rectangle((1,y0),1.0,.48,facecolor=plt.cm.Set2(h),edgecolor='black'))
    ax.text(1.5,y0+.24,f'head {h+1}',ha='center',va='center')
    for j in range(5):
        ax.add_patch(Circle((3+j*.65,y0+.24),.10+.02*((h+j)%3),facecolor=plt.cm.viridis((h+j)/8),edgecolor='none'))
    ax.add_patch(FancyArrowPatch((2.05,y0+.24),(2.7,y0+.24),arrowstyle='->',mutation_scale=12))
    ax.add_patch(FancyArrowPatch((6.1,y0+.24),(7.0,1.72),arrowstyle='->',mutation_scale=12,alpha=.7))
ax.add_patch(Rectangle((7.05,1.25),1.55,.95,facecolor='white',edgecolor='black',lw=1.2))
ax.text(7.82,1.73,'concat + W_O',ha='center',va='center')
ax.set_xlim(.5,9); ax.set_ylim(.2,3.7)
save(fig,'fig11_05_multihead')

# 6. Position encodings heatmaps
fig, axs=plt.subplots(1,3,figsize=(11,3.2))
L=64; d=32
pos=np.arange(L)[:,None]; i=np.arange(d)[None,:]
sin=np.zeros((L,d)); sin[:,0::2]=np.sin(pos/(10000**(i[:,0::2]/d))); sin[:,1::2]=np.cos(pos/(10000**(i[:,1::2]/d)))
rel=np.exp(-np.abs(np.arange(L)[:,None]-np.arange(L)[None,:])/10)
learned=rng.normal(size=(L,d)); learned=np.cumsum(learned,axis=0); learned/=np.std(learned)
for ax,data,title in zip(axs,[sin,learned,rel],['sinusoidal','learned (illustrative)','relative bias kernel']):
    im=ax.imshow(data,aspect='auto',cmap='coolwarm'); ax.set_title(title); ax.set_xlabel('feature / key'); ax.set_ylabel('position / query')
fig.tight_layout(); save(fig,'fig11_06_position')

# 7. Window and shifted window
fig, axs=plt.subplots(1,2,figsize=(8,4))
for ax,shift,title in zip(axs,[0,2],['window attention','shifted-window attention']):
    ax.set_aspect('equal'); ax.set_xlim(0,8); ax.set_ylim(0,8); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
    for i in range(8):
        for j in range(8): ax.add_patch(Rectangle((j,i),1,1,facecolor='white',edgecolor='0.8',lw=.4))
    for i in range(-shift,8,4):
        for j in range(-shift,8,4): ax.add_patch(Rectangle((j,i),4,4,fill=False,edgecolor='tab:blue',lw=2))
    ax.set_title(title)
save(fig,'fig11_07_windows')

# 8. Hierarchical pyramid
fig, ax=plt.subplots(figsize=(10,4.2)); ax.set_axis_off()
levels=[(1.0,0.5,5.5,2.6,64),(3.4,0.8,4.2,2.0,128),(5.7,1.08,2.9,1.45,256),(7.8,1.3,1.8,1.0,512)]
for idx,(x0,y0,w,h,c) in enumerate(levels):
    ax.add_patch(Rectangle((x0,y0),w,h,facecolor=plt.cm.Blues(.25+.15*idx),edgecolor='black'))
    ax.text(x0+w/2,y0+h/2,f'stage {idx+1}\nspatial / {2**idx}\nchannels {c}',ha='center',va='center',fontsize=9)
    if idx<len(levels)-1: ax.add_patch(FancyArrowPatch((x0+w,y0+h/2),(levels[idx+1][0],levels[idx+1][1]+levels[idx+1][3]/2),arrowstyle='->',mutation_scale=15))
ax.set_xlim(.5,10.2); ax.set_ylim(.2,3.7)
save(fig,'fig11_08_hierarchy')

# 9. Attention complexity families
fig, ax=plt.subplots(figsize=(7.5,4.5))
N=np.geomspace(64,4096,50)
ax.loglog(N,N**2,label='global O(N^2)')
ax.loglog(N,49*N,label='window O(M^2 N), M=7')
ax.loglog(N,8*N*np.log2(N),label='hierarchical / sparse (illustrative)')
ax.loglog(N,64*N,label='linearized O(rN), r=64')
ax.set_xlabel('number of tokens N'); ax.set_ylabel('relative interaction cost'); ax.grid(alpha=.3,which='both'); ax.legend()
save(fig,'fig11_09_complexity')

# 10. SSL taxonomy
fig, ax=plt.subplots(figsize=(12,5)); ax.set_axis_off()
centers={'contrastive':(1.5,3.6),'distill':(4.2,3.6),'redundancy':(6.9,3.6),'masked':(9.6,3.6),'predictive':(4.8,1.2),'multimodal':(8.2,1.2)}
texts={'contrastive':'SimCLR\nMoCo\nInfoNCE','distill':'BYOL\nDINO\nDINOv2/3','redundancy':'Barlow Twins\nVICReg','masked':'BEiT\nMAE\niBOT','predictive':'I-JEPA\ndata2vec','multimodal':'CLIP\nSigLIP\nSigLIP 2'}
for k,(x0,y0) in centers.items():
    ax.add_patch(Rectangle((x0-.9,y0-.55),1.8,1.1,facecolor='white',edgecolor='tab:blue',lw=1.3))
    ax.text(x0,y0,texts[k],ha='center',va='center',fontsize=10)
ax.text(5.7,4.8,'self-supervised visual representation learning',ha='center',fontsize=14)
for x0,y0 in centers.values(): ax.add_patch(FancyArrowPatch((5.7,4.55),(x0,y0+.6),arrowstyle='->',mutation_scale=12,alpha=.5))
ax.set_xlim(0,11.2); ax.set_ylim(.3,5.1)
save(fig,'fig11_10_ssl_taxonomy')

# 11. Contrastive geometry
fig, axs=plt.subplots(1,2,figsize=(9,4))
for ax,temp in zip(axs,[0.07,0.5]):
    centers=np.array([[1,0],[-.5,.86],[-.5,-.86]])
    for c in centers:
        pts=c+rng.normal(scale=.12 if temp<.1 else .25,size=(25,2))
        ax.scatter(pts[:,0],pts[:,1],s=14)
    ax.set_aspect('equal'); ax.set_title(f'temperature={temp}'); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('lower temperature sharpens relative similarity penalties')
save(fig,'fig11_11_contrastive')

# 12. Temperature gradients
fig, ax=plt.subplots(figsize=(7,4.2))
s=np.linspace(-1,1,300)
for tau in [.05,.1,.2,.5]:
    p=1/(1+np.exp(-s/tau)); grad=(p-1)/tau
    ax.plot(s,grad,label=f'tau={tau}')
ax.axhline(0,color='black',lw=.5); ax.set_xlabel('positive-minus-negative logit margin'); ax.set_ylabel('illustrative derivative'); ax.grid(alpha=.3); ax.legend()
save(fig,'fig11_12_temperature')

# 13. Augmentation views
fig, axs=plt.subplots(1,4,figsize=(10,2.7))
base=np.zeros((64,64,3)); yy,xx=np.mgrid[:64,:64]; base[...,0]=xx/63; base[...,1]=yy/63; base[...,2]=((xx-32)**2+(yy-32)**2<500)
views=[base,base[8:56,4:60],np.clip(base**.55,0,1),np.flip(base,axis=1)]
views[1]=np.array(plt.cm.viridis(np.mean(views[1],axis=2)))[...,:3]
for ax,img,title in zip(axs,views,['original','crop/color','gamma','flip']): ax.imshow(img); ax.set_title(title); ax.axis('off')
save(fig,'fig11_13_augmentations')

# 14. Collapse diagnostics
fig, axs=plt.subplots(1,2,figsize=(9,4))
eig_good=np.exp(-np.arange(64)/18); eig_bad=np.r_[1,np.full(63,1e-4)]
axs[0].semilogy(eig_good,marker='.'); axs[0].set_title('healthy spectrum'); axs[0].set_xlabel('component'); axs[0].set_ylabel('covariance eigenvalue'); axs[0].grid(alpha=.3)
axs[1].semilogy(eig_bad,marker='.'); axs[1].set_title('collapsed representation'); axs[1].set_xlabel('component'); axs[1].grid(alpha=.3)
save(fig,'fig11_14_collapse')

# 15. EMA schedule
fig, ax=plt.subplots(figsize=(7,4))
t=np.linspace(0,1,200); m=.996+(1-.996)*(1-np.cos(np.pi*t))/2
ax.plot(t,m); ax.set_xlabel('training progress'); ax.set_ylabel('teacher momentum'); ax.set_ylim(.9955,1.0002); ax.grid(alpha=.3)
save(fig,'fig11_15_ema')

# 16. MAE masking
fig, axs=plt.subplots(1,3,figsize=(9,3.1))
grid=rng.uniform(size=(14,14)); mask=rng.uniform(size=(14,14))<.75
vis=grid.copy(); vis[mask]=np.nan
rec=grid.copy(); rec[mask]=.6*grid[mask]+.4*rng.uniform(size=mask.sum())
for ax,data,title in zip(axs,[grid,vis,rec],['target patches','75% masked','decoder reconstruction']): ax.imshow(data,cmap='magma',vmin=0,vmax=1); ax.set_title(title); ax.axis('off')
save(fig,'fig11_16_mae')

# 17. Target types comparison
fig, ax=plt.subplots(figsize=(11,4)); ax.set_axis_off()
items=[('BEiT','discrete codebook token'),('MAE','normalized pixels'),('iBOT','online teacher token'),('I-JEPA','latent target block')]
for i,(a,b) in enumerate(items):
    x0=.7+i*2.55
    ax.add_patch(Rectangle((x0,.8),2.1,2.0,facecolor='white',edgecolor=plt.cm.tab10(i),lw=1.5))
    ax.text(x0+1.05,2.3,a,ha='center',fontsize=14,fontweight='bold'); ax.text(x0+1.05,1.45,b,ha='center',va='center',wrap=True,fontsize=9)
    ax.add_patch(FancyArrowPatch((x0+1.05,.72),(x0+1.05,.25),arrowstyle='->',mutation_scale=13))
ax.text(5.7,.05,'same masking idea, different prediction space and invariance bias',ha='center',fontsize=11)
ax.set_xlim(.2,11); ax.set_ylim(0,3.2)
save(fig,'fig11_17_targets')

# 18. Teacher-student DINO
fig, ax=plt.subplots(figsize=(11,4.5)); ax.set_axis_off()
for x0,title,col in [(1.0,'student','tab:blue'),(7.2,'teacher (EMA)','tab:orange')]:
    ax.add_patch(Rectangle((x0,1.1),2.25,1.7,facecolor='white',edgecolor=col,lw=1.6))
    ax.text(x0+1.125,2.42,title,ha='center',fontsize=12)
    ax.text(x0+1.125,1.75,'encoder + projection\nhead',ha='center',va='center')
ax.text(.9,3.55,'global/local crops',fontsize=10); ax.add_patch(FancyArrowPatch((1.7,3.35),(2.0,2.85),arrowstyle='->',mutation_scale=14))
ax.text(8.0,3.55,'global crops',fontsize=10); ax.add_patch(FancyArrowPatch((8.35,3.35),(8.35,2.85),arrowstyle='->',mutation_scale=14))
ax.add_patch(FancyArrowPatch((3.25,1.95),(7.15,1.95),arrowstyle='<->',mutation_scale=15,lw=1.2))
ax.text(5.2,2.18,'cross-entropy between views',ha='center',fontsize=10)
ax.add_patch(FancyArrowPatch((3.1,.95),(7.3,.95),arrowstyle='->',mutation_scale=15,linestyle='--'))
ax.text(5.2,.72,'EMA parameter update; no gradient',ha='center',fontsize=10)
ax.set_xlim(.4,10.2); ax.set_ylim(.4,4)
save(fig,'fig11_18_dino')

# 19. Native resolution packing
fig, ax=plt.subplots(figsize=(11,4.2)); ax.set_axis_off()
shapes=[(1.0,.6,1.3,2.6),(2.6,.6,2.0,1.35),(4.9,.6,.9,2.2),(6.2,.6,1.8,1.7)]
for i,(x0,y0,w,h) in enumerate(shapes):
    ax.add_patch(Rectangle((x0,y0),w,h,facecolor=plt.cm.Set3(i),edgecolor='black'))
    ax.text(x0+w/2,y0+h/2,f'image {i+1}',ha='center',va='center')
ax.add_patch(FancyArrowPatch((8.3,1.8),(9.0,1.8),arrowstyle='->',mutation_scale=15))
start=9.2
lengths=[8,10,6,9]
colors=[plt.cm.Set3(i) for i in range(4)]
for i,L in enumerate(lengths):
    for j in range(L): ax.add_patch(Rectangle((start+.22*j,.55+.63*i),.19,.45,facecolor=colors[i],edgecolor='none'))
ax.text(10.3,3.45,'packed variable-length token sequences',ha='center',fontsize=11)
ax.set_xlim(.5,12); ax.set_ylim(.3,3.8)
save(fig,'fig11_19_native_pack')

# 20. Token pruning/merging
fig, axs=plt.subplots(1,3,figsize=(11,3.2))
pts=rng.uniform(size=(80,2))
for ax,title,mode in zip(axs,['original tokens','dynamic pruning','token merging'],[0,1,2]):
    if mode==0: p=pts; sizes=np.full(80,15)
    elif mode==1:
        score=np.linalg.norm(pts-[.5,.5],axis=1); keep=score<.48; p=pts[keep]; sizes=np.full(keep.sum(),18)
    else:
        idx=np.argsort(pts[:,0]); groups=np.array_split(idx,28); p=np.array([pts[g].mean(0) for g in groups]); sizes=np.array([18+4*len(g) for g in groups])
    ax.scatter(p[:,0],p[:,1],s=sizes); ax.set_title(title); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0,1); ax.set_ylim(0,1)
save(fig,'fig11_20_token_efficiency')

# 21. Layer attention distance
fig, ax=plt.subplots(figsize=(7.5,4.2))
layers=np.arange(1,13); for_heads=[]
for h in range(6):
    vals=.08+.055*layers+rng.normal(0,.025,size=len(layers))+.03*h
    ax.plot(layers,vals,alpha=.65)
ax.plot(layers,.08+.055*layers+.075,lw=3,label='head average')
ax.set_xlabel('transformer layer'); ax.set_ylabel('normalized attention distance'); ax.grid(alpha=.3); ax.legend()
save(fig,'fig11_21_attention_distance')

# 22. Linear probe vs fine-tune
fig, ax=plt.subplots(figsize=(7.5,4.2))
labels=np.array([1,2,5,10,25,100]); lp=45+12*np.log10(labels); ft=38+18*np.log10(labels); scratch=20+25*np.log10(labels)
ax.semilogx(labels,lp,marker='o',label='linear probe'); ax.semilogx(labels,ft,marker='s',label='fine-tune'); ax.semilogx(labels,scratch,marker='^',label='from scratch')
ax.set_xlabel('labeled data (%)'); ax.set_ylabel('illustrative downstream score'); ax.grid(alpha=.3); ax.legend()
save(fig,'fig11_22_transfer')

# 23. Library stack
fig, ax=plt.subplots(figsize=(11,4.8)); ax.set_axis_off()
layers=[('experiments','Lightning / Accelerate / MMEngine / W&B or MLflow'),('SSL recipes','Lightly / solo-learn / MMSelfSup / official DINO-MAE repos'),('models','timm / torchvision / Hugging Face Transformers'),('kernels','PyTorch SDPA / FlashAttention / xFormers'),('runtime','CUDA / ROCm / CPU / ONNX Runtime / TensorRT')]
for i,(a,b) in enumerate(layers):
    y=.45+i*.8
    ax.add_patch(Rectangle((1,y),9.2,.58,facecolor=plt.cm.Blues(.25+.12*i),edgecolor='black'))
    ax.text(2.2,y+.29,a,ha='center',va='center',fontweight='bold'); ax.text(6.1,y+.29,b,ha='center',va='center',fontsize=9)
ax.set_xlim(.5,10.7); ax.set_ylim(.2,4.8)
save(fig,'fig11_23_libraries')

# 24. Evaluation matrix
fig, ax=plt.subplots(figsize=(8,4.8))
metrics=np.array([[.9,.6,.7,.5,.4],[.8,.85,.8,.7,.65],[.7,.9,.9,.85,.75],[.84,.88,.94,.92,.8]])
im=ax.imshow(metrics,cmap='viridis',vmin=0,vmax=1)
ax.set_xticks(range(5),['linear','kNN','dense','retrieval','OOD'])
ax.set_yticks(range(4),['supervised','contrastive','masked','distillation'])
for i in range(4):
    for j in range(5): ax.text(j,i,f'{metrics[i,j]:.2f}',ha='center',va='center',color='white' if metrics[i,j]<.5 else 'black',fontsize=8)
fig.colorbar(im,ax=ax,label='illustrative normalized quality')
ax.set_title('one benchmark cannot characterize a representation')
save(fig,'fig11_24_evaluation')

# Save numeric tables used in plots
np.savetxt(OUT/'token_cost.csv',np.c_[ps,N[:len(ps)] if len(N)>=len(ps) else np.nan*np.ones(len(ps))],delimiter=',',header='patch_size,token_count',comments='')
with open(OUT/'figure_manifest.txt','w',encoding='utf-8') as f:
    for pth in sorted(FIG.glob('*')): f.write(pth.name+'\n')
print(f'generated {len(list(FIG.glob("*.pdf")))} PDF figures and {len(list(FIG.glob("*.png")))} PNG figures')
