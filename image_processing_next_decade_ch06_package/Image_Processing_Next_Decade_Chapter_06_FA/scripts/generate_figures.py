from __future__ import annotations

from pathlib import Path
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyArrowPatch
import cv2

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


def rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], float)


def roty(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], float)


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], float)


def project(K, R, t, X):
    Xc = (R @ X.T + t.reshape(3,1)).T
    q = (K @ Xc.T).T
    return q[:,:2] / q[:,2:3], Xc[:,2]


def pinhole_geometry():
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    C = np.array([0.0, 0.0])
    img_x = 2.4
    scene_x = 6.4
    P = np.array([scene_x, 2.2])
    Q = np.array([scene_x, -1.45])
    for Y in [P, Q]:
        ax.plot([C[0], Y[0]], [C[1], Y[1]], lw=1.8)
        yi = Y[1] * img_x / Y[0]
        ax.scatter([img_x], [yi], s=36)
        ax.plot([img_x, img_x], [-2.1, 2.1], lw=2.0)
    ax.scatter([0], [0], s=70, marker='o')
    ax.text(-0.25, -0.35, 'Camera center $C$', ha='right')
    ax.text(img_x+0.1, 2.15, 'virtual image plane $Z=f$', va='bottom')
    ax.scatter([P[0], Q[0]], [P[1], Q[1]], s=55)
    ax.text(P[0]+0.1, P[1], '$X_1$')
    ax.text(Q[0]+0.1, Q[1], '$X_2$')
    ax.annotate('', xy=(img_x, -2.55), xytext=(0, -2.55), arrowprops=dict(arrowstyle='<->'))
    ax.text(img_x/2, -2.78, 'focal length $f$', ha='center')
    ax.axhline(0, lw=0.8)
    ax.set_xlim(-0.8, 7.3); ax.set_ylim(-3.0, 3.0)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('Central perspective projection: every pixel represents a ray')
    save(fig, 'pinhole_projection')


def coordinate_pipeline():
    fig, ax = plt.subplots(figsize=(9.3, 4.2))
    ax.axis('off')
    boxes = [
        (0.02,0.35,0.18,0.32,'World frame\\$\\mathbf X_w$'),
        (0.27,0.35,0.18,0.32,'Camera frame\\$\\mathbf X_c=R\\mathbf X_w+t$'),
        (0.52,0.35,0.18,0.32,'Normalized plane\\$(x_n,y_n)$'),
        (0.77,0.35,0.20,0.32,'Pixel coordinates\\$\\mathbf u=K\\mathbf x_n$'),
    ]
    for x,y,w,h,txt in boxes:
        ax.add_patch(plt.Rectangle((x,y),w,h,fill=False,lw=1.8,transform=ax.transAxes))
        ax.text(x+w/2,y+h/2,txt,ha='center',va='center',transform=ax.transAxes)
    for x in [0.20,0.45,0.70]:
        ax.add_patch(FancyArrowPatch((x+0.005,0.51),(x+0.06,0.51),arrowstyle='-|>',mutation_scale=14,transform=ax.transAxes))
    ax.text(0.235,0.62,'extrinsics',ha='center',transform=ax.transAxes)
    ax.text(0.485,0.62,'division by $Z_c$',ha='center',transform=ax.transAxes)
    ax.text(0.735,0.62,'intrinsics + distortion',ha='center',transform=ax.transAxes)
    ax.text(0.5,0.12,'A camera model is a composition of coordinate transforms, a projection law, and a sampling convention.',ha='center',transform=ax.transAxes)
    save(fig, 'coordinate_pipeline')


def camera_projection_lattice():
    K = np.array([[820, 0, 640],[0,820,360],[0,0,1]], float)
    gx, gy = np.meshgrid(np.linspace(-2.5,2.5,9), np.linspace(-1.7,1.7,7))
    X = np.c_[gx.ravel(), gy.ravel(), 7 + 0.35*gx.ravel()]
    poses = [
        (np.eye(3), np.array([0,0,0.])),
        (roty(np.deg2rad(18)), np.array([0.3,-0.1,0.2])),
        (rotx(np.deg2rad(-15))@rotz(np.deg2rad(8)), np.array([-0.3,0.3,0.4]))
    ]
    fig, axes = plt.subplots(1,3,figsize=(11.5,3.5))
    for ax,(R,t),title in zip(axes,poses,['fronto-parallel','yawed camera','tilted camera']):
        uv,z = project(K,R,t,X)
        ax.scatter(uv[:,0],uv[:,1],s=15)
        for j in range(7):
            p=uv[j*9:(j+1)*9]
            ax.plot(p[:,0],p[:,1],lw=.8)
        for i in range(9):
            p=uv[i::9]
            ax.plot(p[:,0],p[:,1],lw=.8)
        ax.set_xlim(250,1030); ax.set_ylim(680,40); ax.set_aspect('equal')
        ax.set_title(title); ax.set_xlabel('u [pixel]'); ax.set_ylabel('v [pixel]')
    save(fig,'perspective_foreshortening')


def distort_grid(kind, k1=0.0, k2=0.0, p1=0.0, p2=0.0):
    vals = np.linspace(-0.9,0.9,19)
    curves=[]
    for c in vals:
        x=np.linspace(-0.95,0.95,250); y=np.full_like(x,c)
        r2=x*x+y*y
        xd=x*(1+k1*r2+k2*r2*r2)+2*p1*x*y+p2*(r2+2*x*x)
        yd=y*(1+k1*r2+k2*r2*r2)+p1*(r2+2*y*y)+2*p2*x*y
        curves.append((xd,yd))
        y=np.linspace(-0.95,0.95,250); x=np.full_like(y,c)
        r2=x*x+y*y
        xd=x*(1+k1*r2+k2*r2*r2)+2*p1*x*y+p2*(r2+2*x*x)
        yd=y*(1+k1*r2+k2*r2*r2)+p1*(r2+2*y*y)+2*p2*x*y
        curves.append((xd,yd))
    return curves


def distortion_models():
    specs=[('ideal',0,0,0,0),('barrel',-0.28,0.05,0,0),('pincushion',0.22,0.04,0,0),('decentering',-0.08,0.02,0.035,-0.02)]
    fig,axes=plt.subplots(1,4,figsize=(13.2,3.2))
    for ax,(name,k1,k2,p1,p2) in zip(axes,specs):
        for x,y in distort_grid(name,k1,k2,p1,p2): ax.plot(x,y,lw=.45)
        ax.set_aspect('equal'); ax.set_xlim(-1.25,1.25); ax.set_ylim(-1.25,1.25)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(name)
    save(fig,'distortion_models')


def fisheye_projection_laws():
    theta=np.linspace(0,np.deg2rad(175),500)
    f=1.0
    laws={
        'perspective $f\\tan\\theta$': np.where(theta<np.deg2rad(89.5), f*np.tan(theta), np.nan),
        'equidistant $f\\theta$': f*theta,
        'equisolid $2f\\sin(\\theta/2)$': 2*f*np.sin(theta/2),
        'stereographic $2f\\tan(\\theta/2)$': 2*f*np.tan(theta/2),
        'orthographic $f\\sin\\theta$': f*np.sin(theta),
    }
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    for label,r in laws.items(): ax.plot(np.rad2deg(theta),r,label=label,lw=1.7)
    ax.set_ylim(0,6); ax.set_xlim(0,175); ax.set_xlabel('incident angle $\\theta$ [degree]'); ax.set_ylabel('image radius $r/f$')
    ax.grid(alpha=.25); ax.legend(fontsize=8); ax.set_title('Projection laws for wide-angle and fisheye cameras')
    save(fig,'fisheye_projection_laws')


def synthetic_calibration(seed=17):
    rng=np.random.default_rng(seed)
    image_size=(1280,720)
    K_true=np.array([[910.,0,638.],[0,895.,362.],[0,0,1.]])
    dist_true=np.array([-0.18,0.055,0.0015,-0.0010,-0.008],float)
    cols,rows=9,6; square=0.035
    obj=np.zeros((rows*cols,3),np.float32)
    obj[:,:2]=np.mgrid[0:cols,0:rows].T.reshape(-1,2)*square
    obj_list=[]; img_list=[]; noiseless=[]; poses=[]
    for i in range(12):
        angles=np.deg2rad([rng.uniform(-20,20),rng.uniform(-25,25),rng.uniform(-12,12)])
        R=rotz(angles[2])@roty(angles[1])@rotx(angles[0])
        rvec,_=cv2.Rodrigues(R)
        tvec=np.array([[rng.uniform(-.12,.10)],[rng.uniform(-.10,.08)],[rng.uniform(.65,1.15)]],float)
        uv,_=cv2.projectPoints(obj,rvec,tvec,K_true,dist_true)
        uv0=uv.reshape(-1,2)
        noisy=uv0+rng.normal(0,.35,uv0.shape)
        if noisy[:,0].min()>20 and noisy[:,0].max()<1260 and noisy[:,1].min()>20 and noisy[:,1].max()<700:
            obj_list.append(obj.copy()); img_list.append(noisy.astype(np.float32)); noiseless.append(uv0); poses.append((rvec,tvec))
    flags=0
    rms,K_est,dist_est,rvecs,tvecs=cv2.calibrateCamera(obj_list,img_list,image_size,None,None,flags=flags)
    per=[]
    all_res=[]
    for i,(op,ip,rv,tv) in enumerate(zip(obj_list,img_list,rvecs,tvecs)):
        pred,_=cv2.projectPoints(op,rv,tv,K_est,dist_est)
        res=np.linalg.norm(pred.reshape(-1,2)-ip,axis=1)
        per.append((i,float(np.sqrt(np.mean(res**2))),float(np.mean(res)),float(np.max(res))))
        all_res.extend(res.tolist())
    with open(OUT/'calibration_summary.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['parameter','true','estimated','absolute_error'])
        names=['fx','fy','cx','cy','k1','k2','p1','p2','k3']
        tv=[K_true[0,0],K_true[1,1],K_true[0,2],K_true[1,2],*dist_true.tolist()]
        ev=[K_est[0,0],K_est[1,1],K_est[0,2],K_est[1,2],*dist_est.ravel()[:5].tolist()]
        for n,a,b in zip(names,tv,ev):w.writerow([n,a,b,abs(a-b)])
        w.writerow(['global_rms','',rms,''])
    with open(OUT/'calibration_per_view.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['view','rmse_px','mean_px','max_px']); w.writerows(per)

    fig,axes=plt.subplots(2,3,figsize=(11.2,7.0))
    for ax,ip in zip(axes.ravel(),img_list[:6]):
        ax.scatter(ip[:,0],ip[:,1],s=8)
        ax.set_xlim(0,image_size[0]);ax.set_ylim(image_size[1],0);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Synthetic planar calibration observations (diverse pose and image coverage)')
    save(fig,'calibration_views')

    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    axes[0].bar(np.arange(len(per)),[p[1] for p in per]); axes[0].set_xlabel('view');axes[0].set_ylabel('reprojection RMSE [px]');axes[0].set_title(f'global RMS = {rms:.3f} px')
    axes[1].hist(all_res,bins=22);axes[1].set_xlabel('residual magnitude [px]');axes[1].set_ylabel('count');axes[1].set_title('residual distribution')
    save(fig,'calibration_residuals')
    return K_true,K_est,dist_true,dist_est,image_size


def epipolar_experiment(seed=41):
    rng=np.random.default_rng(seed)
    K=np.array([[900.,0,640.],[0,900.,360.],[0,0,1.]])
    R1=np.eye(3); t1=np.zeros(3)
    R2=roty(np.deg2rad(8))@rotx(np.deg2rad(-2)); C2=np.array([0.55,0.02,0.04]); t2=-R2@C2
    X=np.c_[rng.uniform(-1.8,1.8,80),rng.uniform(-1.1,1.1,80),rng.uniform(4.5,9.0,80)]
    u1,_=project(K,R1,t1,X);u2,_=project(K,R2,t2,X)
    n1=u1+rng.normal(0,.55,u1.shape);n2=u2+rng.normal(0,.55,u2.shape)
    F,_=cv2.findFundamentalMat(n1.astype(np.float32),n2.astype(np.float32),cv2.FM_8POINT)
    # normalize rank 2
    U,S,Vt=np.linalg.svd(F);S[-1]=0;F=U@np.diag(S)@Vt
    x1=np.c_[n1,np.ones(len(n1))];x2=np.c_[n2,np.ones(len(n2))]
    Fx1=(F@x1.T).T;Ftx2=(F.T@x2.T).T
    num=np.sum(x2*Fx1,axis=1)**2
    den=Fx1[:,0]**2+Fx1[:,1]**2+Ftx2[:,0]**2+Ftx2[:,1]**2
    sampson=num/den
    np.savetxt(OUT/'estimated_fundamental_matrix.csv',F,delimiter=',')
    with open(OUT/'epipolar_metrics.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['metric','value']);w.writerow(['mean_sampson_px2',float(np.mean(sampson))]);w.writerow(['median_sampson_px2',float(np.median(sampson))]);w.writerow(['rank',int(np.linalg.matrix_rank(F))])
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.4))
    axes[0].scatter(n1[:,0],n1[:,1],s=12)
    axes[0].set_xlim(0,1280);axes[0].set_ylim(720,0);axes[0].set_title('view 1 correspondences')
    axes[1].scatter(n2[:,0],n2[:,1],s=12)
    inds=np.linspace(0,len(n1)-1,12,dtype=int)
    lines=(F@x1[inds].T).T
    xs=np.array([0,1280])
    for l in lines:
        ys=-(l[0]*xs+l[2])/l[1]
        axes[1].plot(xs,ys,lw=.8)
    axes[1].set_xlim(0,1280);axes[1].set_ylim(720,0);axes[1].set_title('view 2 and epipolar lines')
    for ax in axes:ax.set_xlabel('u [px]');ax.set_ylabel('v [px]')
    save(fig,'epipolar_lines')


def stereo_uncertainty():
    f=900.; Bvals=[.08,.20,.50]; sigma_d=.25
    Z=np.linspace(1,25,400)
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.1))
    for B in Bvals:
        d=f*B/Z
        sigma_z=Z**2/(f*B)*sigma_d
        axes[0].plot(Z,d,label=f'B={B:.2f} m')
        axes[1].plot(Z,sigma_z,label=f'B={B:.2f} m')
    axes[0].set_xlabel('depth Z [m]');axes[0].set_ylabel('disparity d [px]');axes[0].set_title('$d=fB/Z$')
    axes[1].set_xlabel('depth Z [m]');axes[1].set_ylabel('$\\sigma_Z$ [m]');axes[1].set_title('first-order depth uncertainty')
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8)
    save(fig,'stereo_depth_uncertainty')


def rolling_shutter():
    h,w=220,320
    y=np.arange(h)
    # vertical world lines, row-dependent horizontal displacement and shear
    fig,axes=plt.subplots(1,3,figsize=(11.7,4.0))
    xlines=np.linspace(35,w-35,7)
    for x0 in xlines:
        axes[0].plot(np.full_like(y,x0),y,lw=1.3)
        xlin=x0+0.22*(y-h/2)
        axes[1].plot(xlin,y,lw=1.3)
        xrot=(x0-w/2)*np.cos(.0025*(y-h/2))-(y-h/2)*np.sin(.0025*(y-h/2))+w/2
        axes[2].plot(xrot,y,lw=1.3)
    titles=['global shutter','RS: constant translation','RS: row-varying rotation']
    for ax,t in zip(axes,titles):
        ax.set_xlim(0,w);ax.set_ylim(h,0);ax.set_aspect('equal');ax.set_title(t);ax.set_xticks([]);ax.set_yticks([])
    save(fig,'rolling_shutter_geometry')


def bundle_graph():
    rng=np.random.default_rng(4)
    fig,ax=plt.subplots(figsize=(8.6,4.6));ax.axis('off')
    cams=np.c_[np.linspace(.08,.92,7),np.full(7,.78)]
    pts=np.c_[rng.uniform(.08,.92,16),rng.uniform(.12,.38,16)]
    for i,c in enumerate(cams):
        ax.add_patch(plt.Rectangle((c[0]-.025,c[1]-.035),.05,.07,fill=False,lw=1.4,transform=ax.transAxes));ax.text(c[0],c[1]+.06,f'$C_{i+1}$',ha='center',transform=ax.transAxes)
    for j,p in enumerate(pts):
        ax.add_patch(Circle((p[0],p[1]),.012,fill=False,lw=1.2,transform=ax.transAxes));
        if j<8:ax.text(p[0],p[1]-.035,f'$X_{j+1}$',ha='center',fontsize=8,transform=ax.transAxes)
    for i,c in enumerate(cams):
        seen=rng.choice(len(pts),size=rng.integers(6,11),replace=False)
        for j in seen:
            p=pts[j];ax.plot([c[0],p[0]],[c[1],p[1]],lw=.35,alpha=.45,transform=ax.transAxes)
    ax.text(.5,.94,'Sparse camera-point observation graph',ha='center',fontsize=12,transform=ax.transAxes)
    ax.text(.5,.02,'Schur complement eliminates point increments and preserves the sparse camera system.',ha='center',transform=ax.transAxes)
    save(fig,'bundle_adjustment_graph')


def nerf_camera_rays():
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.6))
    ax=axes[0]
    C=np.array([0.,0.]); plane_x=1.6
    pix_y=np.linspace(-1.2,1.2,9)
    for py in pix_y:
        ax.plot([C[0],4.8],[C[1],py*4.8/plane_x],lw=.7,alpha=.75)
        ts=np.linspace(2.0,4.2,7);ys=py*ts/plane_x
        ax.scatter(ts,ys,s=8)
    ax.plot([plane_x,plane_x],[-1.4,1.4],lw=2)
    ax.scatter([0],[0],s=45);ax.set_aspect('equal');ax.set_xlim(-.5,5);ax.set_ylim(-3,3);ax.axis('off');ax.set_title('camera rays and differentiable volume samples')
    ax=axes[1]
    # projected covariance ellipses / gaussian splats
    rng=np.random.default_rng(2)
    for i in range(18):
        x,y=rng.uniform(-2.6,2.6),rng.uniform(-1.8,1.8)
        a,b=rng.uniform(.12,.45),rng.uniform(.08,.28);ang=rng.uniform(0,np.pi)
        tt=np.linspace(0,2*np.pi,100);xx=a*np.cos(tt);yy=b*np.sin(tt)
        R=np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])
        q=R@np.vstack([xx,yy]);ax.fill(x+q[0],y+q[1],alpha=.22);ax.plot(x+q[0],y+q[1],lw=.5)
    ax.set_aspect('equal');ax.set_xlim(-3,3);ax.set_ylim(-2.2,2.2);ax.set_xticks([]);ax.set_yticks([]);ax.set_title('projected anisotropic 3D Gaussian footprints')
    save(fig,'differentiable_camera_rays')


def main():
    pinhole_geometry();coordinate_pipeline();camera_projection_lattice();distortion_models();fisheye_projection_laws()
    synthetic_calibration();epipolar_experiment();stereo_uncertainty();rolling_shutter();bundle_graph();nerf_camera_rays()
    print('Generated figures in',FIG)

if __name__=='__main__':main()
