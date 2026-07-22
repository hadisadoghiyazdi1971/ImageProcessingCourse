from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, Ellipse
from scipy.ndimage import gaussian_filter, rotate, shift
from scipy.signal import correlate2d
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve, roc_curve, auc

OUT = Path(__file__).resolve().parent / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.png', dpi=220, bbox_inches='tight')
    plt.close(fig)

# 1. Pipeline diagram
fig, ax = plt.subplots(figsize=(11, 3.1))
ax.axis('off')
labels = ['تصویر', 'آشکارسازی', 'توصیف', 'تطبیق', 'راستی‌آزمایی هندسی', 'بازشناسی/بازیابی']
xs = np.linspace(0.05, 0.95, len(labels))
for i, (x, lab) in enumerate(zip(xs, labels)):
    w = 0.13 if i not in (4,5) else 0.16
    ax.add_patch(Rectangle((x-w/2, 0.38), w, 0.24, transform=ax.transAxes,
                           facecolor='white', edgecolor='black', lw=1.5))
    ax.text(x, 0.5, lab, ha='center', va='center', transform=ax.transAxes, fontsize=11)
    if i < len(labels)-1:
        ax.add_patch(FancyArrowPatch((x+w/2,0.5),(xs[i+1]-(0.13 if i+1 not in (4,5) else 0.16)/2,0.5),
                                     transform=ax.transAxes, arrowstyle='-|>', mutation_scale=12, lw=1.4))
ax.text(0.5, 0.12, 'هر مرحله باید ناوردایی، دقت مکانی، تمایزبخشی و هزینه محاسباتی خود را توجیه کند.',
        ha='center', transform=ax.transAxes, fontsize=10)
save(fig, 'fig09_01_pipeline')

# synthetic image utility
n=256
y,x=np.mgrid[0:n,0:n]
img=np.zeros((n,n),float)
img += 0.8*((x>35)&(x<120)&(y>40)&(y<125))
img += 0.65*((x-180)**2+(y-70)**2<30**2)
img += 0.9*((np.abs(x-y-10)<4)&(x>80)&(x<220))
img += 0.25*np.sin(x/6)*np.cos(y/9)
img = gaussian_filter(img,1.2)
img=(img-img.min())/(img.max()-img.min())

# 2 Harris geometry/eigenvalue plane
Ix=np.gradient(img,axis=1); Iy=np.gradient(img,axis=0)
Sxx=gaussian_filter(Ix*Ix,2); Syy=gaussian_filter(Iy*Iy,2); Sxy=gaussian_filter(Ix*Iy,2)
det=Sxx*Syy-Sxy*Sxy; tr=Sxx+Syy
R=det-0.04*tr*tr
thr=np.quantile(R,0.997)
pts=np.argwhere(R>thr)
fig, axs=plt.subplots(1,2,figsize=(10,4))
axs[0].imshow(img,cmap='gray'); axs[0].scatter(pts[:,1],pts[:,0],s=18,facecolors='none',edgecolors='r'); axs[0].set_title('پاسخ هریس و نقاط منتخب'); axs[0].axis('off')
lam1=np.linspace(0,1,200); lam2=np.linspace(0,1,200)
L1,L2=np.meshgrid(lam1,lam2); RR=L1*L2-0.04*(L1+L2)**2
cs=axs[1].contourf(L1,L2,RR,levels=20)
axs[1].plot([0,1],[0,0],lw=1); axs[1].plot([0,0],[0,1],lw=1)
axs[1].set_xlabel(r'$\lambda_1$'); axs[1].set_ylabel(r'$\lambda_2$'); axs[1].set_title('صفحه مقادیر ویژه: تخت، لبه، گوشه')
fig.colorbar(cs,ax=axs[1],shrink=.8)
save(fig,'fig09_02_harris')

# 3 scale-space and DoG
sigmas=[1,2,4,8]
blurs=[gaussian_filter(img,s) for s in sigmas]
fig, axs=plt.subplots(2,4,figsize=(12,6))
for j,(s,b) in enumerate(zip(sigmas,blurs)):
    axs[0,j].imshow(b,cmap='gray'); axs[0,j].set_title(fr'$\sigma={s}$'); axs[0,j].axis('off')
for j in range(3):
    dog=blurs[j+1]-blurs[j]
    axs[1,j].imshow(dog,cmap='coolwarm'); axs[1,j].set_title(f'DoG {sigmas[j]}→{sigmas[j+1]}'); axs[1,j].axis('off')
axs[1,3].axis('off')
axs[1,3].text(.5,.55,'اکسترمم سه‌بعدی\nدر $x,y,\sigma$',ha='center',va='center',fontsize=15)
save(fig,'fig09_03_scale_space')

# 4 SIFT descriptor visualization
fig, ax=plt.subplots(figsize=(6,6))
ax.set_xlim(0,4); ax.set_ylim(0,4); ax.set_aspect('equal'); ax.invert_yaxis()
for i in range(5): ax.plot([i,i],[0,4],color='0.7',lw=.8)
for j in range(5): ax.plot([0,4],[j,j],color='0.7',lw=.8)
for iy in range(4):
    for ix in range(4):
        angles=np.linspace(0,2*np.pi,8,endpoint=False)
        mags=np.abs(np.sin(angles*2 + 0.7*ix - 0.4*iy))+0.15
        mags=mags/mags.max()*0.36
        cx,cy=ix+.5,iy+.5
        for a,m in zip(angles,mags):
            ax.arrow(cx,cy,m*np.cos(a),m*np.sin(a),head_width=.04,head_length=.05,length_includes_head=True,lw=.8)
ax.set_xticks([]); ax.set_yticks([]); ax.set_title('توصیفگر SIFT: ۴×۴ سلول و ۸ سبد جهت')
save(fig,'fig09_04_sift_descriptor')

# 5 binary tests
fig, ax=plt.subplots(figsize=(7,5))
ax.imshow(img,cmap='gray'); ax.axis('off')
center=np.array([128,128])
for k in range(18):
    p=center+rng.normal(0,35,2)
    q=center+rng.normal(0,35,2)
    ax.plot([p[0],q[0]],[p[1],q[1]],lw=1)
    ax.scatter([p[0],q[0]],[p[1],q[1]],s=10)
ax.add_patch(Circle(center,55,fill=False,lw=1.5))
ax.set_title('آزمون‌های دودویی BRIEF/ORB: مقایسه شدت زوج نقاط')
save(fig,'fig09_05_binary_tests')

# 6 descriptor distance and ratio test
nq=160
true_d=rng.normal(.35,.08,nq).clip(.05,1)
second_d=(true_d+rng.normal(.22,.09,nq)).clip(.08,1.25)
false1=rng.normal(.68,.12,nq).clip(.1,1.3)
false2=(false1+rng.normal(.07,.06,nq)).clip(.1,1.4)
r_true=true_d/second_d; r_false=false1/false2
fig, axs=plt.subplots(1,2,figsize=(10,4))
axs[0].scatter(true_d,second_d,s=14,label='تطابق درست',alpha=.75)
axs[0].scatter(false1,false2,s=14,label='تطابق نادرست',alpha=.75)
xx=np.linspace(0,1.3,100); axs[0].plot(xx,xx/.8,'k--',label=r'$d_1/d_2=0.8$')
axs[0].set_xlabel(r'$d_1$'); axs[0].set_ylabel(r'$d_2$'); axs[0].legend(fontsize=8); axs[0].set_title('آزمون نسبت همسایه نزدیک')
axs[1].hist(r_true,bins=25,alpha=.65,label='درست'); axs[1].hist(r_false,bins=25,alpha=.65,label='نادرست')
axs[1].axvline(.8,ls='--',color='k'); axs[1].set_xlabel(r'$d_1/d_2$'); axs[1].legend(); axs[1].set_title('تفکیک آماری نسبت فاصله')
save(fig,'fig09_06_ratio_test')

# 7 RANSAC synthetic line
xv=np.linspace(0,10,90); yv=1.7*xv+2+rng.normal(0,.55,len(xv))
outx=rng.uniform(0,10,40); outy=rng.uniform(-2,23,40)
X=np.r_[xv,outx]; Y=np.r_[yv,outy]
best=(0,None)
for _ in range(800):
    ids=rng.choice(len(X),2,replace=False)
    if abs(X[ids[1]]-X[ids[0]])<1e-6: continue
    a=(Y[ids[1]]-Y[ids[0]])/(X[ids[1]]-X[ids[0]]); b=Y[ids[0]]-a*X[ids[0]]
    res=np.abs(a*X-Y+b)/np.sqrt(a*a+1)
    mask=res<.8
    if mask.sum()>best[0]: best=(mask.sum(),(a,b,mask))
a,b,mask=best[1]
fig, ax=plt.subplots(figsize=(7,5))
ax.scatter(X[~mask],Y[~mask],s=20,label='برون‌ریز')
ax.scatter(X[mask],Y[mask],s=20,label='درون‌ریز')
ax.plot(xx:=np.linspace(0,10,100),a*xx+b,'k',lw=2,label='مدل RANSAC')
ax.legend(); ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title('برازش مقاوم و اجماع هندسی')
save(fig,'fig09_07_ransac')

# 8 homography verification schematic
fig, axs=plt.subplots(1,2,figsize=(11,4.5))
for ax in axs:
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal'); ax.axis('off')
pts1=rng.uniform(.1,.9,(35,2))
H=np.array([[1.0,.18,-.08],[-.08,.95,.08],[.08,-.05,1]])
hp=np.c_[pts1,np.ones(len(pts1))]@H.T; pts2=hp[:,:2]/hp[:,2:]
pts2+=rng.normal(0,.006,pts2.shape)
pts2[-10:]=rng.uniform(.05,.95,(10,2))
axs[0].scatter(pts1[:,0],pts1[:,1],s=18); axs[0].set_title('تصویر مرجع')
axs[1].scatter(pts2[:,0],pts2[:,1],s=18); axs[1].set_title('تصویر هدف')
for i in range(len(pts1)):
    con=FancyArrowPatch((pts1[i,0],pts1[i,1]),(pts2[i,0]+1.1,pts2[i,1]),transform=axs[0].transData,
                        arrowstyle='-',lw=.4,alpha=.35)
# instead draw on combined hidden axes
fig.suptitle('راستی‌آزمایی هندسی: تطابق توصیفگر کافی نیست')
save(fig,'fig09_08_geometric_verification')

# 9 BoVW pipeline
fig, ax=plt.subplots(figsize=(11,3.6)); ax.axis('off')
labels=['توصیفگرهای محلی','خوشه‌بندی k-means','واژگان بصری','هیستوگرام تصویر','طبقه‌بند/بازیابی']
xs=np.linspace(.08,.92,len(labels))
for i,(x0,l) in enumerate(zip(xs,labels)):
    ax.add_patch(Rectangle((x0-.085,.36),.17,.28,transform=ax.transAxes,facecolor='white',edgecolor='black'))
    ax.text(x0,.5,l,ha='center',va='center',transform=ax.transAxes,fontsize=10)
    if i<len(labels)-1:
        ax.add_patch(FancyArrowPatch((x0+.085,.5),(xs[i+1]-.085,.5),transform=ax.transAxes,arrowstyle='-|>',mutation_scale=12))
for k in range(40):
    ax.scatter(.02+.12*rng.random(),.15+.15*rng.random(),s=8,transform=ax.transAxes)
ax.text(.5,.12,'کدگذاری، pooling، نرمال‌سازی و هندسه مکانی کیفیت نمایش نهایی را تعیین می‌کنند.',ha='center',transform=ax.transAxes)
save(fig,'fig09_09_bovw')

# 10 spatial pyramid
fig, axs=plt.subplots(1,3,figsize=(10,3.4))
levels=[1,2,4]
for ax,L in zip(axs,levels):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal'); ax.axis('off')
    for i in range(L+1):
        ax.plot([i/L,i/L],[0,1],'k',lw=1)
        ax.plot([0,1],[i/L,i/L],'k',lw=1)
    p=rng.uniform(0,1,(35,2)); ax.scatter(p[:,0],p[:,1],s=12)
    ax.set_title(f'سطح {int(np.log2(L))}: {L}×{L}')
fig.suptitle('هرم مکانی: بازگرداندن آرایش تقریبی به کیسه کلمات')
save(fig,'fig09_10_spatial_pyramid')

# 11 PCA / LDA geometry
c1=rng.multivariate_normal([-1,0],[[1.2,.8],[.8,1.1]],120)
c2=rng.multivariate_normal([1.2,.3],[[1.0,.7],[.7,1.0]],120)
Xall=np.r_[c1,c2]; ylab=np.r_[np.zeros(len(c1)),np.ones(len(c2))]
pca=PCA(2).fit(Xall); vp=pca.components_[0]
mu1=c1.mean(0); mu2=c2.mean(0); Sw=np.cov(c1,rowvar=False)+np.cov(c2,rowvar=False); vlda=np.linalg.solve(Sw,mu2-mu1); vlda/=np.linalg.norm(vlda)
fig, ax=plt.subplots(figsize=(7,5))
ax.scatter(c1[:,0],c1[:,1],s=12,alpha=.5,label='کلاس ۱'); ax.scatter(c2[:,0],c2[:,1],s=12,alpha=.5,label='کلاس ۲')
origin=Xall.mean(0)
ax.arrow(origin[0],origin[1],2*vp[0],2*vp[1],head_width=.12,length_includes_head=True,label='PCA')
ax.arrow(origin[0],origin[1],2*vlda[0],2*vlda[1],head_width=.12,length_includes_head=True)
ax.text(*(origin+2.05*vp),'PCA'); ax.text(*(origin+2.05*vlda),'LDA')
ax.legend(); ax.set_title('PCA بیشینه واریانس؛ LDA بیشینه جدایی کلاس')
save(fig,'fig09_11_pca_lda')

# 12 SVM margin
c1=rng.normal([-1.2,0],.55,(90,2)); c2=rng.normal([1.2,.25],.55,(90,2)); X=np.r_[c1,c2]; y=np.r_[-np.ones(90),np.ones(90)]
clf=SVC(kernel='linear',C=1).fit(X,y)
xx,yy=np.meshgrid(np.linspace(-3,3,250),np.linspace(-2.5,2.5,250)); Z=clf.decision_function(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
fig, ax=plt.subplots(figsize=(7,5))
ax.scatter(c1[:,0],c1[:,1],s=14,label='-1'); ax.scatter(c2[:,0],c2[:,1],s=14,label='+1')
ax.contour(xx,yy,Z,levels=[-1,0,1],linestyles=['--','-','--'])
sv=clf.support_vectors_; ax.scatter(sv[:,0],sv[:,1],s=70,facecolors='none',edgecolors='k',label='بردار پشتیبان')
ax.legend(); ax.set_title('ماشین بردار پشتیبان و حاشیه هندسی')
save(fig,'fig09_12_svm')

# 13 inverted index / tfidf schematic
fig, ax=plt.subplots(figsize=(10,4.5)); ax.axis('off')
words=['w1','w2','w3','w4','w5']; docs=['I1','I2','I3','I4']
for i,w in enumerate(words):
    ax.text(.08,.85-i*.16,w,ha='center',va='center',fontsize=12,bbox=dict(boxstyle='round',fc='white'))
postings={0:[0,2],1:[0,1,3],2:[1],3:[0,2,3],4:[2,3]}
for i,w in enumerate(words):
    x0=.18
    for j,d in enumerate(postings[i]):
        ax.text(x0+j*.18,.85-i*.16,f'{docs[d]}: tf={rng.integers(1,6)}',ha='center',va='center',fontsize=9,bbox=dict(boxstyle='round',fc='0.95'))
ax.text(.5,.05,'فهرست معکوس فقط تصاویر دارای واژه مشترک را بازیابی می‌کند؛ IDF واژه‌های نادر را تقویت می‌کند.',ha='center')
ax.set_title('بازیابی مقیاس‌پذیر با TF-IDF و فهرست معکوس')
save(fig,'fig09_13_inverted_index')

# 14 PR/ROC and threshold
scores_pos=rng.beta(5,2,500); scores_neg=rng.beta(2,5,800)
ytrue=np.r_[np.ones(len(scores_pos)),np.zeros(len(scores_neg))]; scores=np.r_[scores_pos,scores_neg]
prec,rec,thr=precision_recall_curve(ytrue,scores); fpr,tpr,_=roc_curve(ytrue,scores)
fig, axs=plt.subplots(1,2,figsize=(10,4))
axs[0].plot(rec,prec); axs[0].set_xlabel('Recall'); axs[0].set_ylabel('Precision'); axs[0].set_title(f'PR curve, AP≈{np.trapz(prec[::-1],rec[::-1]):.3f}')
axs[1].plot(fpr,tpr); axs[1].plot([0,1],[0,1],'k--'); axs[1].set_xlabel('FPR'); axs[1].set_ylabel('TPR'); axs[1].set_title(f'ROC, AUC={auc(fpr,tpr):.3f}')
save(fig,'fig09_14_metrics')

# 15 classical-modern spectrum
methods=['Harris+patch','SIFT','ORB','AKAZE','SuperPoint','LightGlue','LoFTR','RoMa','RDD']
repeat=np.array([.58,.84,.68,.76,.86,.90,.89,.91,.93])
speed=np.array([.95,.45,.92,.78,.72,.82,.55,.25,.48])
robust=np.array([.38,.76,.54,.70,.80,.88,.91,.94,.92])
Y=np.arange(len(methods))
fig, ax=plt.subplots(figsize=(8,5.5))
ax.scatter(repeat,Y,s=55,label='تکرارپذیری/دقت'); ax.scatter(speed,Y,s=55,label='سرعت نسبی'); ax.scatter(robust,Y,s=55,label='تاب‌آوری تغییر دید')
ax.set_yticks(Y,methods); ax.set_xlim(0,1); ax.grid(axis='x',alpha=.3); ax.legend(fontsize=8); ax.set_title('طیف روش‌ها؛ مقادیر صرفاً آموزشی و غیرمعیار رسمی‌اند')
save(fig,'fig09_15_classic_modern')

print(f'Generated figures in {OUT}')
