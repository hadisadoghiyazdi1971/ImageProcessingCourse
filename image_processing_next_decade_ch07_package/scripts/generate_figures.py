from __future__ import annotations

from pathlib import Path
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from scipy import ndimage as ndi
from skimage import draw, filters, feature, morphology, segmentation, measure, util

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
OUT = ROOT / "outputs"
FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 140,
})


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def neighborhoods_and_connectivity() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.7))
    for ax, conn, title in zip(axes[:2], [1, 2], ["4-neighborhood", "8-neighborhood"]):
        ax.set_aspect("equal")
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
        for x in range(-2, 3):
            for y in range(-2, 3):
                ax.add_patch(Rectangle((x-.42,y-.42),.84,.84,fill=False,lw=.7))
        ax.add_patch(Rectangle((-.42,-.42),.84,.84,alpha=.35))
        offsets=[(1,0),(-1,0),(0,1),(0,-1)]
        if conn==2: offsets += [(1,1),(1,-1),(-1,1),(-1,-1)]
        for x,y in offsets: ax.add_patch(Rectangle((x-.42,y-.42),.84,.84,alpha=.22))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)
    ax=axes[2]
    pattern=np.array([[1,0,1],[0,1,0],[1,0,1]])
    ax.imshow(pattern,cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
    for i in range(4): ax.axhline(i-.5,lw=.6); ax.axvline(i-.5,lw=.6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title("Digital-topology ambiguity")
    ax.text(1,3.05,"Foreground and background cannot both use 8-connectivity",ha='center',fontsize=9)
    save(fig,"neighborhood_connectivity")


def synthetic_local_image(n=256):
    y,x=np.mgrid[:n,:n]
    I=np.zeros((n,n),float)
    I[:, :70]=0.12
    I[:,70:125]=0.60
    I[35:120,135:220]=0.85
    # smooth ridge
    ridge=np.exp(-((y-185-10*np.sin(x/25))**2)/(2*3.2**2))*0.9
    I=np.maximum(I,ridge)
    # blob
    I += .65*np.exp(-((x-205)**2+(y-185)**2)/(2*15**2))
    I=ndi.gaussian_filter(I,1.2)
    return np.clip(I,0,1)


def local_differential_structure() -> None:
    I=synthetic_local_image()
    L=ndi.gaussian_filter(I,1.5)
    Ix=ndi.gaussian_filter(I,1.5,order=(0,1))
    Iy=ndi.gaussian_filter(I,1.5,order=(1,0))
    g=np.hypot(Ix,Iy)
    Axy=feature.structure_tensor(I,sigma=2.2,order='rc')
    eig=feature.structure_tensor_eigenvalues(Axy)
    l1,l2=eig[0],eig[1]
    coherence=(l1-l2)/(l1+l2+1e-12)
    Hxx=ndi.gaussian_filter(I,2.0,order=(0,2))
    Hxy=ndi.gaussian_filter(I,2.0,order=(1,1))
    Hyy=ndi.gaussian_filter(I,2.0,order=(2,0))
    detH=Hxx*Hyy-Hxy**2
    fig,axes=plt.subplots(2,3,figsize=(11.2,7.2))
    ims=[I,L,g,l1+l2,coherence,np.abs(detH)]
    titles=['synthetic local structures','Gaussian scale-space','gradient magnitude','structure-tensor energy','tensor coherence','absolute Hessian determinant']
    for ax,z,t in zip(axes.ravel(),ims,titles):
        im=ax.imshow(z,cmap='gray'); ax.set_title(t); ax.set_xticks([]);ax.set_yticks([])
        fig.colorbar(im,ax=ax,fraction=.046,pad=.02)
    save(fig,'local_differential_structure')
    flat=(l1+l2<np.percentile(l1+l2,35))
    edge=(coherence>.75)&(~flat)
    corner=(l2>np.percentile(l2,98.5))
    with open(OUT/'local_structure_summary.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['class','pixel_count','fraction'])
        for name,mask in [('flat',flat),('edge_like',edge),('corner_like',corner)]:
            w.writerow([name,int(mask.sum()),float(mask.mean())])


def tensor_eigenvalue_plane() -> None:
    rng=np.random.default_rng(7)
    fig,ax=plt.subplots(figsize=(6.2,5.2))
    groups={
        'flat':np.abs(rng.normal([.03,.02],[.018,.012],(160,2))),
        'edge':np.c_[rng.uniform(.4,1.0,160),rng.uniform(.0,.08,160)],
        'corner':rng.uniform(.35,1.0,(160,2)),
    }
    for label,v in groups.items():
        v=np.sort(v,axis=1)[:,::-1]
        ax.scatter(v[:,0],v[:,1],s=12,alpha=.55,label=label)
    ax.plot([0,1.05],[0,1.05],ls='--',lw=1)
    ax.set_xlim(0,1.05);ax.set_ylim(0,1.05)
    ax.set_xlabel('$\\lambda_1$');ax.set_ylabel('$\\lambda_2$')
    ax.legend();ax.grid(alpha=.2);ax.set_title('Local structure in the eigenvalue plane')
    save(fig,'structure_tensor_eigenplane')


def binary_test_image(n=220):
    X=np.zeros((n,n),bool)
    rr,cc=draw.disk((72,65),34,shape=X.shape);X[rr,cc]=1
    X[42:102,115:175]=1
    X[65:80,175:202]=1
    X[145:190,35:185]=1
    rr,cc=draw.disk((167,95),12,shape=X.shape);X[rr,cc]=0
    X[155:161,185:204]=1
    return X


def binary_morphology_figure() -> None:
    X=binary_test_image();B=morphology.disk(8)
    imgs=[X,morphology.binary_dilation(X,B),morphology.binary_erosion(X,B),morphology.binary_opening(X,B),morphology.binary_closing(X,B)]
    titles=['input set $X$','dilation $X\\oplus B$','erosion $X\\ominus B$','opening $X\\circ B$','closing $X\\bullet B$']
    fig,axes=plt.subplots(1,5,figsize=(14,3.1))
    for ax,z,t in zip(axes,imgs,titles):ax.imshow(z,cmap='gray');ax.set_title(t);ax.axis('off')
    save(fig,'binary_morphology')
    with open(OUT/'binary_morphology_metrics.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['operation','foreground_area','components','euler_number'])
        for name,z in zip(['input','dilation','erosion','opening','closing'],imgs):
            lab=measure.label(z,connectivity=2)
            w.writerow([name,int(z.sum()),int(lab.max()),float(measure.euler_number(z,connectivity=2))])


def grayscale_profiles() -> None:
    x=np.linspace(0,1,500)
    f=.25+.12*np.sin(2*np.pi*3*x)+.55*np.exp(-(x-.34)**2/(2*.018**2))-.22*np.exp(-(x-.72)**2/(2*.03**2))
    size=41
    d=ndi.maximum_filter1d(f,size=size,mode='nearest')
    e=ndi.minimum_filter1d(f,size=size,mode='nearest')
    op=ndi.maximum_filter1d(e,size=size,mode='nearest')
    cl=ndi.minimum_filter1d(d,size=size,mode='nearest')
    fig,axes=plt.subplots(2,1,figsize=(9.2,5.8),sharex=True)
    axes[0].plot(x,f,label='signal');axes[0].plot(x,d,label='dilation');axes[0].plot(x,e,label='erosion')
    axes[0].legend(ncol=3);axes[0].grid(alpha=.2);axes[0].set_title('Flat grayscale morphology as max/min filtering')
    axes[1].plot(x,f-op,label='white top-hat');axes[1].plot(x,cl-f,label='black top-hat')
    axes[1].axhline(0,lw=.7);axes[1].legend();axes[1].grid(alpha=.2);axes[1].set_xlabel('position')
    save(fig,'grayscale_morphology_profiles')


def reconstruction_figure() -> None:
    n=240
    mask=np.zeros((n,n),float)
    rr,cc=draw.disk((105,85),52,shape=mask.shape);mask[rr,cc]=.75
    rr,cc=draw.disk((110,155),45,shape=mask.shape);mask[rr,cc]=.75
    mask[45:65,75:165]=.75
    mask[145:178,112:128]=.75
    mask += .22*np.exp(-(((np.arange(n)[:,None]-70)**2+(np.arange(n)[None,:]-180)**2)/(2*18**2)))
    mask=ndi.gaussian_filter(mask,1.0)
    er=morphology.erosion(mask,morphology.disk(13))
    opened=morphology.opening(mask,morphology.disk(13))
    rec=morphology.reconstruction(er,mask,method='dilation')
    hmax=morphology.h_maxima(mask,.10)
    fig,axes=plt.subplots(1,5,figsize=(14,3.1))
    for ax,z,t in zip(axes,[mask,er,opened,rec,hmax],['mask image','marker (erosion)','ordinary opening','opening by reconstruction','$h$-maxima']):
        ax.imshow(z,cmap='gray');ax.set_title(t);ax.axis('off')
    save(fig,'morphological_reconstruction')


def component_tree_figure() -> None:
    img=np.array([
        [0,0,0,0,0,0,0,0,0],
        [0,2,2,1,1,1,3,3,0],
        [0,2,5,1,4,1,3,6,0],
        [0,2,2,1,4,1,3,3,0],
        [0,0,0,1,1,1,0,0,0],
    ],float)
    fig,axes=plt.subplots(1,2,figsize=(9.2,4.0))
    axes[0].imshow(img,cmap='gray',vmin=0,vmax=6,interpolation='nearest')
    for i in range(img.shape[0]+1):axes[0].axhline(i-.5,lw=.5)
    for j in range(img.shape[1]+1):axes[0].axvline(j-.5,lw=.5)
    for (i,j),v in np.ndenumerate(img):axes[0].text(j,i,str(int(v)),ha='center',va='center',fontsize=8)
    axes[0].axis('off');axes[0].set_title('gray-level image')
    ax=axes[1];ax.axis('off');ax.set_xlim(0,10);ax.set_ylim(0,7)
    nodes={0:(5,.7,'level 0'),1:(5,2.0,'level 1'),2:(2.3,3.2,'level 2'),3:(7.7,3.2,'level 3'),4:(5,4.35,'level 4'),5:(2.3,5.5,'level 5'),6:(7.7,5.5,'level 6')}
    edges=[(0,1),(1,2),(1,3),(1,4),(2,5),(3,6)]
    for a,b in edges:
        xa,ya,_=nodes[a];xb,yb,_=nodes[b];ax.plot([xa,xb],[ya,yb],lw=1.5)
    for k,(x,y,label) in nodes.items():
        ax.add_patch(Circle((x,y),.43,fill=False,lw=1.5));ax.text(x,y,str(k),ha='center',va='center')
        ax.text(x+.55,y,label,va='center',fontsize=8)
    ax.set_title('schematic max-tree: inclusion of upper-level components')
    save(fig,'component_tree_schematic')


def dt_1d(f: np.ndarray) -> np.ndarray:
    """Felzenszwalb-Huttenlocher squared distance transform in O(n)."""
    n=len(f);v=np.zeros(n,dtype=int);z=np.zeros(n+1,float);d=np.zeros(n,float)
    k=0;v[0]=0;z[0]=-np.inf;z[1]=np.inf
    for q in range(1,n):
        s=((f[q]+q*q)-(f[v[k]]+v[k]*v[k]))/(2*q-2*v[k])
        while s<=z[k]:
            k-=1
            s=((f[q]+q*q)-(f[v[k]]+v[k]*v[k]))/(2*q-2*v[k])
        k+=1;v[k]=q;z[k]=s;z[k+1]=np.inf
    k=0
    for q in range(n):
        while z[k+1]<q:k+=1
        d[q]=(q-v[k])**2+f[v[k]]
    return d


def distance_transform_figure() -> None:
    n=130
    sites=[18,52,92,116]
    f=np.full(n,1e6);f[sites]=[0,8,2,12]
    d=dt_1d(f)
    x=np.arange(n)
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.2))
    for q in sites:axes[0].plot(x,(x-q)**2+f[q],lw=.9,alpha=.6)
    axes[0].plot(x,d,lw=2.4,label='lower envelope')
    axes[0].set_ylim(0,850);axes[0].set_xlabel('q');axes[0].set_ylabel('cost');axes[0].legend();axes[0].set_title('1-D exact squared distance transform')
    X=binary_test_image(220)
    D=ndi.distance_transform_edt(X)
    im=axes[1].imshow(D,cmap='magma');axes[1].contour(D,levels=8,linewidths=.7)
    axes[1].axis('off');axes[1].set_title('Euclidean distance inside a binary set')
    fig.colorbar(im,ax=axes[1],fraction=.046,pad=.03)
    save(fig,'distance_transform_parabolas')
    sizes=[128,256,512,1024]
    rows=[]
    rng=np.random.default_rng(22)
    for m in sizes:
        arr=rng.random((m,m))>.72
        t0=time.perf_counter();ndi.distance_transform_edt(arr);dt=time.perf_counter()-t0
        rows.append((m,m*m,dt))
    with open(OUT/'edt_timing.csv','w',newline='') as fcsv:
        w=csv.writer(fcsv);w.writerow(['side','pixels','scipy_edt_seconds']);w.writerows(rows)


def skeleton_figure() -> None:
    X=binary_test_image(260)
    # connect a branch to create richer medial geometry
    rr,cc=draw.polygon([120,138,220,215],[165,185,220,198],shape=X.shape);X[rr,cc]=1
    D=ndi.distance_transform_edt(X)
    skel,dist=morphology.medial_axis(X,return_distance=True)
    thin=morphology.skeletonize(X)
    fig,axes=plt.subplots(1,4,figsize=(13.1,3.4))
    axes[0].imshow(X,cmap='gray');axes[0].set_title('binary shape')
    im=axes[1].imshow(D,cmap='magma');axes[1].set_title('distance transform');fig.colorbar(im,ax=axes[1],fraction=.046,pad=.02)
    axes[2].imshow(X,cmap='gray');axes[2].imshow(np.ma.masked_where(~skel,skel),cmap='autumn');axes[2].set_title('medial axis')
    axes[3].imshow(X,cmap='gray');axes[3].imshow(np.ma.masked_where(~thin,thin),cmap='autumn');axes[3].set_title('topological thinning')
    for ax in axes:ax.axis('off')
    save(fig,'skeleton_medial_axis')
    with open(OUT/'skeleton_metrics.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['representation','pixels','endpoints_approx'])
        for name,S in [('medial_axis',skel),('skeletonize',thin)]:
            neigh=ndi.convolve(S.astype(int),np.ones((3,3),int),mode='constant')-S
            w.writerow([name,int(S.sum()),int(np.sum(S&(neigh==1)))])


def watershed_figure() -> None:
    n=260;y,x=np.mgrid[:n,:n]
    centers=[(92,92,46),(105,145,50),(170,118,52),(175,178,38)]
    img=np.zeros((n,n),float)
    truth=np.zeros((n,n),int)
    for i,(cy,cx,r) in enumerate(centers,1):
        z=np.exp(-((x-cx)**2+(y-cy)**2)/(2*(r*.62)**2))
        img+=z
        truth[((x-cx)**2+(y-cy)**2)<r*r]=i
    rng=np.random.default_rng(14);img=ndi.gaussian_filter(img+rng.normal(0,.035,img.shape),1.2)
    mask=img>.22
    grad=filters.sobel(img)
    local_max=feature.peak_local_max(img,min_distance=28,labels=mask)
    markers=np.zeros_like(truth)
    for i,(r,c) in enumerate(local_max,1):markers[r,c]=i
    markers=measure.label(markers>0)
    ws=segmentation.watershed(grad,markers,mask=mask)
    naive=segmentation.watershed(grad,mask=mask)
    fig,axes=plt.subplots(1,5,figsize=(14.3,3.2))
    entries=[(img,'touching objects'),(grad,'gradient topography'),(naive,'uncontrolled watershed'),(markers,'markers'),(ws,'marker-controlled watershed')]
    for ax,(z,t) in zip(axes,entries):ax.imshow(z,cmap='nipy_spectral' if 'watershed' in t or t=='markers' else 'gray');ax.set_title(t);ax.axis('off')
    save(fig,'watershed_marker_controlled')
    with open(OUT/'watershed_metrics.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['method','regions','foreground_pixels'])
        w.writerow(['uncontrolled',int(naive.max()),int((naive>0).sum())]);w.writerow(['marker_controlled',int(ws.max()),int((ws>0).sum())])


def granulometry_figure() -> None:
    n=300;X=np.zeros((n,n),bool)
    rng=np.random.default_rng(10)
    radii=[4,6,8,10,13,17,22,28]
    for r in radii:
        for _ in range(4):
            cy=rng.integers(r+3,n-r-3);cx=rng.integers(r+3,n-r-3)
            rr,cc=draw.disk((cy,cx),r,shape=X.shape);X[rr,cc]=1
    scales=np.arange(1,36)
    area=[]
    for s in scales:area.append(morphology.opening(X,morphology.disk(int(s))).sum())
    area=np.asarray(area,float);spectrum=np.maximum(0,-np.diff(np.r_[area,area[-1]]))
    fig,axes=plt.subplots(1,3,figsize=(12.2,3.7))
    axes[0].imshow(X,cmap='gray');axes[0].axis('off');axes[0].set_title('objects at multiple sizes')
    axes[1].plot(scales,area/area[0]);axes[1].set_xlabel('structuring-element radius');axes[1].set_ylabel('remaining area ratio');axes[1].grid(alpha=.2);axes[1].set_title('granulometry')
    axes[2].bar(scales,spectrum/(spectrum.sum()+1e-12));axes[2].set_xlabel('radius');axes[2].set_ylabel('normalized pattern spectrum');axes[2].set_title('morphological size distribution')
    save(fig,'granulometry_pattern_spectrum')
    with open(OUT/'granulometry.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['radius','remaining_area','pattern_spectrum']);w.writerows(zip(scales,area,spectrum))


def softmax_beta(v,beta):
    m=np.max(beta*v)
    return (m+np.log(np.sum(np.exp(beta*v-m))))/beta


def soft_morphology_figure() -> None:
    x=np.linspace(0,1,360)
    sig=.2+.15*np.sin(2*np.pi*4*x)+.65*np.exp(-(x-.52)**2/(2*.022**2))
    half=13;betas=[1,4,12,40]
    exact=ndi.maximum_filter1d(sig,size=2*half+1,mode='nearest')
    approx=[]
    padded=np.pad(sig,half,mode='edge')
    for b in betas:
        z=np.array([softmax_beta(padded[i:i+2*half+1],b) for i in range(len(sig))])
        approx.append(z)
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.1))
    axes[0].plot(x,sig,label='input',lw=1.2);axes[0].plot(x,exact,label='hard dilation',lw=2)
    for b,z in zip(betas,approx):axes[0].plot(x,z,label=f'$\\beta={b}$',alpha=.8)
    axes[0].legend(fontsize=8,ncol=2);axes[0].grid(alpha=.2);axes[0].set_title('LogSumExp relaxation of dilation')
    errors=[np.max(np.abs(z-exact)) for z in approx]
    axes[1].loglog(betas,errors,'o-');axes[1].grid(alpha=.25,which='both');axes[1].set_xlabel('$\\beta$');axes[1].set_ylabel('$\\|\\delta_\\beta f-\\delta f\\|_\\infty$');axes[1].set_title('convergence to max-plus morphology')
    save(fig,'soft_morphology_temperature')
    with open(OUT/'soft_morphology_error.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['beta','max_abs_error']);w.writerows(zip(betas,errors))


def main():
    neighborhoods_and_connectivity()
    local_differential_structure()
    tensor_eigenvalue_plane()
    binary_morphology_figure()
    grayscale_profiles()
    reconstruction_figure()
    component_tree_figure()
    distance_transform_figure()
    skeleton_figure()
    watershed_figure()
    granulometry_figure()
    soft_morphology_figure()
    print(f"Wrote figures to {FIG}")
    print(f"Wrote CSV outputs to {OUT}")


if __name__ == '__main__':
    main()
