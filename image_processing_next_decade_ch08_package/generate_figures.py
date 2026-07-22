from __future__ import annotations

from pathlib import Path
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from scipy import ndimage as ndi
from sklearn.mixture import GaussianMixture
from skimage import data, color, exposure, filters, measure, morphology, segmentation, transform, util
from skimage.filters import threshold_otsu, threshold_multiotsu, threshold_sauvola
from skimage.segmentation import random_walker, slic, morphological_chan_vese

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
OUT = ROOT / "outputs"
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(1405)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def synthetic_scene(n: int = 256):
    yy, xx = np.mgrid[:n, :n]
    clean = 0.15 + 0.2 * xx / n + 0.15 * yy / n
    mask1 = (xx - 82) ** 2 + (yy - 95) ** 2 < 43 ** 2
    mask2 = ((xx - 177) / 52) ** 2 + ((yy - 155) / 31) ** 2 < 1
    mask3 = (xx > 135) & (xx < 220) & (yy > 36) & (yy < 78)
    clean = clean.copy()
    clean[mask1] += 0.43
    clean[mask2] += 0.28
    clean[mask3] += 0.52
    truth = np.zeros((n, n), dtype=np.uint8)
    truth[mask1] = 1
    truth[mask2] = 2
    truth[mask3] = 3
    vignette = 0.38 * ((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / (n / 2) ** 2
    observed = np.clip(clean - vignette + RNG.normal(0, 0.07, clean.shape), 0, 1)
    return clean, observed, truth


def dice(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    d = a.sum() + b.sum()
    return 1.0 if d == 0 else 2.0 * np.logical_and(a, b).sum() / d


def iou(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else np.logical_and(a, b).sum() / u


def fig_taxonomy():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    boxes = {
        "Input image": (0.05, .42),
        "Binary / multi-class": (.28, .76),
        "Semantic": (.28, .55),
        "Instance": (.28, .34),
        "Panoptic": (.28, .13),
        "Interactive / promptable": (.60, .76),
        "Video / 3-D": (.60, .50),
        "Open-vocabulary / concept": (.60, .24),
    }
    for text, (x, y) in boxes.items():
        w = .27 if x > .5 else .25
        patch = FancyBboxPatch((x, y), w, .11, boxstyle="round,pad=0.02", lw=1.5, fill=False)
        ax.add_patch(patch); ax.text(x+w/2, y+.055, text, ha="center", va="center", fontsize=11)
    for target in ["Binary / multi-class", "Semantic", "Instance", "Panoptic"]:
        x1, y1 = boxes["Input image"]; x2, y2 = boxes[target]
        ax.add_patch(FancyArrowPatch((x1+.25, y1+.055), (x2, y2+.055), arrowstyle="->", mutation_scale=12))
    for source, target in [("Semantic","Interactive / promptable"),("Instance","Video / 3-D"),("Panoptic","Open-vocabulary / concept")]:
        x1,y1=boxes[source]; x2,y2=boxes[target]
        ax.add_patch(FancyArrowPatch((x1+.25,y1+.055),(x2,y2+.055),arrowstyle="->",mutation_scale=12))
    ax.text(.5,.96,"Segmentation task taxonomy",ha="center",fontsize=18,weight="bold")
    ax.text(.5,.02,"The output semantics, supervision and interaction protocol define different mathematical tasks.",ha="center",fontsize=10)
    save(fig, "fig01_segmentation_taxonomy")


def fig_otsu():
    _, obs, truth = synthetic_scene()
    vals = util.img_as_ubyte(obs)
    t = threshold_otsu(vals)
    mts = threshold_multiotsu(vals, classes=4)
    hist, bins = np.histogram(vals, bins=256, range=(0,255), density=True)
    centers = (bins[:-1]+bins[1:])/2
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(centers, hist, lw=1.5)
    ax.axvline(t, ls="--", label=f"Otsu t={t}")
    for i,tt in enumerate(mts): ax.axvline(tt, ls=":", label=f"multi-Otsu t{i+1}={tt}")
    ax.set(xlabel="Gray level", ylabel="Probability", title="Global and multi-level thresholding")
    ax.legend(ncol=2, fontsize=9)
    save(fig, "fig02_otsu_histogram")


def fig_adaptive():
    clean, obs, truth = synthetic_scene()
    global_t = threshold_otsu(obs)
    global_mask = obs > global_t
    local_t = threshold_sauvola(obs, window_size=41, k=.18)
    local_mask = obs > local_t
    target = truth > 0
    fig, axs = plt.subplots(1,4,figsize=(14,3.6))
    for ax, im, title in zip(axs,[obs,target,global_mask,local_mask],
                             ["Uneven illumination","Ground truth",f"Global Otsu\nDice={dice(global_mask,target):.3f}",f"Sauvola\nDice={dice(local_mask,target):.3f}"]):
        ax.imshow(im,cmap="gray"); ax.set_title(title); ax.axis("off")
    save(fig, "fig03_adaptive_thresholding")


def region_grow(image, seed, tol):
    h,w=image.shape; out=np.zeros_like(image,dtype=bool); seen=np.zeros_like(out)
    q=[seed]; seen[seed]=True; mean=float(image[seed]); count=1
    while q:
        r,c=q.pop(0)
        if abs(float(image[r,c])-mean)<=tol:
            out[r,c]=True; mean += (float(image[r,c])-mean)/count; count += 1
            for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                rr,cc=r+dr,c+dc
                if 0<=rr<h and 0<=cc<w and not seen[rr,cc]:
                    seen[rr,cc]=True; q.append((rr,cc))
    return out


def fig_region():
    _, obs, truth = synthetic_scene(220)
    seed=(95,82)
    masks=[region_grow(obs,seed,t) for t in (.035,.07,.13)]
    fig,axs=plt.subplots(1,4,figsize=(14,3.5))
    axs[0].imshow(obs,cmap="gray"); axs[0].plot(seed[1],seed[0],"r+"); axs[0].set_title("Seed and image")
    for ax,m,t in zip(axs[1:],masks,(.035,.07,.13)):
        ax.imshow(m,cmap="gray"); ax.set_title(f"Tolerance={t}")
    for ax in axs: ax.axis("off")
    save(fig,"fig04_region_growing")


def fig_gmm():
    clean, obs, truth = synthetic_scene(180)
    yy,xx=np.mgrid[:obs.shape[0],:obs.shape[1]]
    feat=np.column_stack([obs.ravel(), xx.ravel()/obs.shape[1], yy.ravel()/obs.shape[0]])
    idx=RNG.choice(feat.shape[0],size=6500,replace=False)
    gmm=GaussianMixture(4,covariance_type="full",random_state=7).fit(feat[idx])
    lab=gmm.predict(feat).reshape(obs.shape)
    fig,axs=plt.subplots(1,3,figsize=(12,3.7))
    axs[0].imshow(obs,cmap="gray"); axs[0].set_title("Input")
    sc=axs[1].scatter(feat[idx,1],feat[idx,0],c=gmm.predict(feat[idx]),s=3,cmap="tab10")
    axs[1].set(xlabel="x/W",ylabel="intensity",title="GMM feature space")
    axs[2].imshow(lab,cmap="tab10"); axs[2].set_title("Spatial GMM labels")
    for ax in (axs[0],axs[2]): ax.axis("off")
    save(fig,"fig05_gmm_clustering")


def binary_graph_cut(image, fg_seed, bg_seed, beta=40.0, lam=2.0):
    import networkx as nx
    h,w=image.shape
    G=nx.DiGraph(); S="s"; T="t"; eps=1e-6
    fg_mu=float(image[fg_seed]); bg_mu=float(image[bg_seed]); sig=.09
    for r in range(h):
        for c in range(w):
            node=(r,c)
            dfg=(float(image[r,c])-fg_mu)**2/(2*sig**2)
            dbg=(float(image[r,c])-bg_mu)**2/(2*sig**2)
            G.add_edge(S,node,capacity=lam*dbg+eps)
            G.add_edge(node,T,capacity=lam*dfg+eps)
            if (r,c)==fg_seed: G[S][node]["capacity"]=1e6
            if (r,c)==bg_seed: G[node][T]["capacity"]=1e6
            for dr,dc in ((1,0),(0,1)):
                rr,cc=r+dr,c+dc
                if rr<h and cc<w:
                    wij=math.exp(-beta*(float(image[r,c])-float(image[rr,cc]))**2)+eps
                    G.add_edge(node,(rr,cc),capacity=wij); G.add_edge((rr,cc),node,capacity=wij)
    _,part=nx.minimum_cut(G,S,T)
    reachable,_=part
    return np.array([[(r,c) in reachable for c in range(w)] for r in range(h)])


def fig_graph_cut():
    _, obs, truth=synthetic_scene(64)
    fg=(24,21); bg=(2,2)
    mask=binary_graph_cut(obs,fg,bg)
    fig,axs=plt.subplots(1,3,figsize=(10,3.3))
    axs[0].imshow(obs,cmap="gray"); axs[0].plot([fg[1],bg[1]],[fg[0],bg[0]],"rx"); axs[0].set_title("Seeds")
    axs[1].imshow(mask,cmap="gray"); axs[1].set_title("s-t min cut")
    axs[2].imshow(obs,cmap="gray"); axs[2].contour(mask,[.5],linewidths=2); axs[2].set_title("Boundary")
    for ax in axs: ax.axis("off")
    save(fig,"fig06_graph_cut")


def fig_random_walker():
    _,obs,truth=synthetic_scene(128)
    markers=np.zeros(obs.shape,dtype=np.int32)
    markers[12:17,12:17]=1
    markers[45:50,38:43]=2
    labels=random_walker(obs,markers,beta=120,mode="cg_j")
    fig,axs=plt.subplots(1,3,figsize=(10,3.3))
    axs[0].imshow(obs,cmap="gray"); axs[0].imshow(np.ma.masked_where(markers==0,markers),alpha=.8,cmap="coolwarm"); axs[0].set_title("Seed labels")
    axs[1].imshow(labels,cmap="coolwarm"); axs[1].set_title("Random walker")
    axs[2].imshow(obs,cmap="gray"); axs[2].contour(labels==2,[.5]); axs[2].set_title("Harmonic boundary")
    for ax in axs: ax.axis("off")
    save(fig,"fig07_random_walker")


def fig_chan_vese():
    _,obs,truth=synthetic_scene(180)
    init="checkerboard"
    states=[]
    def cb(levelset):
        if len(states)<5: states.append(levelset.copy())
    final=morphological_chan_vese(obs,num_iter=85,init_level_set=init,smoothing=2,iter_callback=cb)
    fig,axs=plt.subplots(1,4,figsize=(13,3.4))
    for ax,im,title in zip(axs,[obs,states[0] if states else final,states[-1] if states else final,final],
                           ["Image","Initialization","Intermediate","Final Chan-Vese"]):
        ax.imshow(obs if title!="Initialization" else im,cmap="gray")
        if title!="Image" and title!="Initialization": ax.contour(im,[.5])
        ax.set_title(title); ax.axis("off")
    save(fig,"fig08_chan_vese")


def fig_slic():
    img=transform.resize(data.coffee(),(240,320),anti_aliasing=True)
    labels=slic(img,n_segments=220,compactness=12,start_label=1,channel_axis=-1)
    avg=color.label2rgb(labels,img,kind="avg")
    boundary=segmentation.mark_boundaries(img,labels)
    fig,axs=plt.subplots(1,3,figsize=(12,3.5))
    for ax,im,title in zip(axs,[img,boundary,avg],["Image","SLIC boundaries","Piecewise-constant approximation"]):
        ax.imshow(im); ax.set_title(title); ax.axis("off")
    save(fig,"fig09_slic_superpixels")


def fig_losses():
    p=np.linspace(.001,.999,500)
    y=1
    ce=-np.log(p)
    focal=-(1-p)**2*np.log(p)
    dice_loss=1-(2*p+1e-6)/(p+1+1e-6)
    tv_loss=1-(p+1e-6)/(p+.7*(1-p)+.3*(1-p)*0+1e-6)
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(p,ce,label="Cross-entropy")
    ax.plot(p,focal,label="Focal (gamma=2)")
    ax.plot(p,dice_loss,label="Soft Dice")
    ax.plot(p,tv_loss,label="Tversky-like")
    ax.set(xlabel="Predicted foreground probability p",ylabel="Loss",title="Loss geometry for a positive pixel",ylim=(0,5))
    ax.legend()
    save(fig,"fig10_loss_geometry")


def fig_unet():
    fig,ax=plt.subplots(figsize=(12,6)); ax.axis("off")
    levels=[(0.05,.72,.15,.14,"H x W\n64"),(0.22,.59,.15,.14,"H/2\n128"),(0.39,.46,.15,.14,"H/4\n256"),(0.56,.33,.15,.14,"H/8\n512")]
    for x,y,w,h,t in levels:
        ax.add_patch(Rectangle((x,y),w,h,fill=False,lw=2)); ax.text(x+w/2,y+h/2,t,ha="center",va="center")
    up=[(0.72,.46,.15,.14,"H/4\n256"),(0.82,.59,.15,.14,"H/2\n128"),(0.87,.72,.11,.14,"H x W\nC")]
    for x,y,w,h,t in up:
        ax.add_patch(Rectangle((x,y),w,h,fill=False,lw=2)); ax.text(x+w/2,y+h/2,t,ha="center",va="center")
    pts=levels+up
    for (a,b) in zip(pts[:-1],pts[1:]):
        ax.add_patch(FancyArrowPatch((a[0]+a[2],a[1]+a[3]/2),(b[0],b[1]+b[3]/2),arrowstyle="->",mutation_scale=14))
    for i,(x,y,w,h,t) in enumerate(levels[:-1]):
        tx,ty,tw,th,_=up[-(i+1)]
        ax.add_patch(FancyArrowPatch((x+w/2,y+h),(tx+tw/2,ty+th),arrowstyle="->",linestyle="--",connectionstyle="arc3,rad=-.25"))
    ax.text(.5,.95,"Encoder-decoder with multi-scale skip connections",ha="center",fontsize=16,weight="bold")
    save(fig,"fig11_unet_architecture")


def fig_metrics():
    n=220; yy,xx=np.mgrid[:n,:n]
    gt=(xx-n/2)**2+(yy-n/2)**2<(58)**2
    pred_shift=(xx-(n/2+9))**2+(yy-n/2)**2<(58)**2
    pred_erode=(xx-n/2)**2+(yy-n/2)**2<(49)**2
    fig,axs=plt.subplots(1,3,figsize=(11,3.6))
    for ax,p,title in zip(axs,[gt,pred_shift,pred_erode],["Ground truth",f"Shifted\nIoU={iou(gt,pred_shift):.3f}",f"Eroded\nIoU={iou(gt,pred_erode):.3f}"]):
        ax.imshow(gt,cmap="gray",alpha=.45); ax.contour(p,[.5]); ax.set_title(title); ax.axis("off")
    save(fig,"fig12_region_boundary_metrics")


def fig_calibration():
    n=150000
    logits=RNG.normal(size=n)
    true_prob=1/(1+np.exp(-logits))
    y=RNG.binomial(1,true_prob)
    over=1/(1+np.exp(-2.2*logits))
    bins=np.linspace(0,1,11)
    def rel(p):
        xs=[]; ys=[]; ns=[]
        for a,b in zip(bins[:-1],bins[1:]):
            m=(p>=a)&(p<b)
            if m.any(): xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(m.sum())
        return np.array(xs),np.array(ys),np.array(ns)
    x1,y1,_=rel(true_prob); x2,y2,_=rel(over)
    fig,ax=plt.subplots(figsize=(6,5))
    ax.plot([0,1],[0,1],"--",label="Perfect calibration")
    ax.plot(x1,y1,"o-",label="Calibrated")
    ax.plot(x2,y2,"s-",label="Over-confident")
    ax.set(xlabel="Mean confidence",ylabel="Empirical frequency",title="Pixelwise reliability diagram")
    ax.legend(); save(fig,"fig13_segmentation_calibration")


def fig_sam_timeline():
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    events = [
        (2023, "SAM", "point / box / mask\npromptable image masks"),
        (2024, "SAM 2", "streaming memory\nimages + video"),
        (2025, "SAM 3", "text / exemplar concepts\ndetect + segment + track"),
    ]
    xs = [.16, .50, .84]
    ax.plot([.1, .9], [.5, .5], lw=2, transform=ax.transAxes, clip_on=False)
    for x, (yr, name, desc) in zip(xs, events):
        ax.scatter([x], [.5], s=180, zorder=3, transform=ax.transAxes, clip_on=False)
        ax.text(x, .70, f"{yr}\n{name}", ha="center", va="center", fontsize=14,
                weight="bold", transform=ax.transAxes)
        ax.text(x, .25, desc, ha="center", va="center", fontsize=10,
                transform=ax.transAxes)
    ax.text(.5, .93, "Evolution of promptable segmentation", ha="center",
            fontsize=17, weight="bold", transform=ax.transAxes)
    save(fig, "fig14_sam_evolution")


def fig_failure_modes():
    n=192; yy,xx=np.mgrid[:n,:n]
    gt=((xx-65)**2+(yy-95)**2<38**2)|(((xx-130)/35)**2+((yy-95)/18)**2<1)
    masks={
        "Boundary bias": ndi.binary_erosion(gt,iterations=5),
        "Fragmentation": gt & ~((xx>87)&(xx<98)),
        "Leakage": gt | ((xx>100)&(xx<145)&(yy>72)&(yy<118)),
        "Small-object miss": ((xx-65)**2+(yy-95)**2<38**2),
    }
    fig,axs=plt.subplots(1,4,figsize=(13,3.3))
    for ax,(name,m) in zip(axs,masks.items()):
        ax.imshow(gt,cmap="gray",alpha=.35); ax.contour(m,[.5]); ax.set_title(name); ax.axis("off")
    save(fig,"fig15_failure_modes")


def write_metrics():
    clean,obs,truth=synthetic_scene()
    target=truth>0
    gt=target
    t=threshold_otsu(obs); gm=obs>t
    sm=obs>threshold_sauvola(obs,window_size=41,k=.18)
    rows=[("global_otsu",float(t),dice(gm,gt),iou(gm,gt)),("sauvola",float(np.mean(threshold_sauvola(obs,41,k=.18))),dice(sm,gt),iou(sm,gt))]
    with (OUT/"threshold_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["method","threshold_summary","dice","iou"]); w.writerows(rows)


def main():
    fig_taxonomy(); fig_otsu(); fig_adaptive(); fig_region(); fig_gmm(); fig_graph_cut(); fig_random_walker(); fig_chan_vese(); fig_slic(); fig_losses(); fig_unet(); fig_metrics(); fig_calibration(); fig_sam_timeline(); fig_failure_modes(); write_metrics()
    print(f"Wrote figures to {FIG}")

if __name__ == "__main__":
    main()
