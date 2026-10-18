"""
Bragg's Law and Absorption of X-rays (UP 201, Experiment 7)
Generates every figure used in the lab report from the raw PHYWE measure data.

Usage:   python make_figures.py        (reads ./data, writes ./figures)
Needs:   numpy, scipy, matplotlib

Assumptions: GM dead time tau = 90 us; rho(Al) = 2.70 g/cm^3; d(LiF) = 201.4 pm;
file 'lif_4mm' is the 0.04 mm Al absorber. Angle uncertainty = half a 0.1 deg step.
"""
import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.3,'figure.dpi':150,'savefig.bbox':'tight'})
U='data/'
def load(f):
    t=open(U+f,encoding='latin-1').read().replace('\r','').split('\n')
    rows=[l.split('\t') for l in t[2:] if l.strip()]
    if f=='0.08': return np.array([[float(r[5]),float(r[1])] for r in rows])
    return np.array([[float(r[0]),float(r[1])] for r in rows])
tau=90e-6; T=2.0
dt=lambda N0: N0/(1-tau*N0)
sdt=lambda N0: np.sqrt(np.maximum(N0,1)/T)/(1-tau*N0)**2
hc=1239.84198  # eV nm
dLiF=201.4
g=lambda x,A,m,s,a,b: A*np.exp(-(x-m)**2/(2*s**2))+a+b*x
def fit(D,lo,hi):
    x=D[:,0]; k=(x>=lo)&(x<=hi); x=x[k]; y=dt(D[k,1]); sg=sdt(D[k,1])
    p0=[y.max()-y.min(),x[np.argmax(y)],0.12,y.min(),0]
    p,c=curve_fit(g,x,y,p0=p0,sigma=sg,absolute_sigma=True,maxfev=50000)
    chi=np.sum(((y-g(x,*p))/sg)**2)/(len(x)-5)
    e=np.sqrt(np.diag(c))*max(1,np.sqrt(chi))
    return p,e,(lo,hi),chi
def save(fig,name):
    fig.savefig(f'figures/{name}.pdf'); fig.savefig(f'figures/{name}.png',dpi=200)
R={}
os.makedirs('figures',exist_ok=True)
L=load('LiF'); K=load('kBr')
# ---------------- LiF -----------------
lif_pk=[('K$_\\beta$',1,19.7,20.8),('K$_\\alpha$',1,22.0,23.2),('K$_\\beta$',2,43.2,44.3),('K$_\\alpha$',2,49.5,50.7)]
lif=[]
for nm,n,lo,hi in lif_pk:
    p,e,r,chi=fit(L,lo,hi)
    th=p[1]; sth=np.hypot(e[1],0.05)
    lam=2*dLiF*np.sin(np.radians(th))/n          # pm
    slam=lam/np.tan(np.radians(th))*np.radians(sth)
    E=hc/lam*1000/1000  # eV nm / pm -> hc/(lam nm)
    E=hc/(lam/1000)     # eV
    sE=E*slam/lam
    lif.append(dict(name=nm,n=n,th=th,sth_fit=e[1],sth=sth,peak=p[0]+p[3]+p[4]*th,fwhm=2.3548*abs(p[2]),lam=lam,slam=slam,E=E/1e3,sE=sE/1e3,
                    raw=float(L[np.argmin(abs(L[:,0]-th)),1])))
    print(lif[-1])
R['lif']=lif
def wmean(v,s): w=1/np.array(s)**2; return np.sum(w*np.array(v))/w.sum(), 1/np.sqrt(w.sum())
for nm in ['K$_\\alpha$','K$_\\beta$']:
    a=[d for d in lif if d['name']==nm]
    lm,sl=wmean([d['lam'] for d in a],[d['slam'] for d in a])
    sd=np.std([d['lam'] for d in a],ddof=1)
    lm_s=np.mean([d['lam'] for d in a])
    # final: unweighted mean, uncertainty = max(std error, weighted)
    sm=max(np.std([d['lam'] for d in a],ddof=1)/np.sqrt(len(a)),sl)
    Em=hc/(lm_s/1000)/1e3; sEm=Em*sm/lm_s
    R[nm]=dict(lam=lm_s,slam=sm,E=Em,sE=sEm,wlam=lm,wsl=sl)
    print(nm,R[nm])
# Duane-Hunt
lmin=hc/35e3*1000  # pm
R['lmin']=lmin; R['thmin']=np.degrees(np.arcsin(lmin/(2*dLiF)))
print('lmin',lmin,R['thmin'])
# ---- LiF figures
fig,ax=plt.subplots(figsize=(9,4.6))
ax.plot(L[:,0],L[:,1],color='0.6',lw=.9,label='Measured (uncorrected)')
ax.plot(L[:,0],dt(L[:,1]),'C3',lw=1.1,label=r'Dead-time corrected ($\tau=90\,\mu$s)')
offs=[(-7,3200),(5,8500),(-5,2200),(-4,2500)]
for d,(dx,ytxt) in zip(lif,offs):
    ax.annotate(f"{d['name']} (n={d['n']}): {d['th']:.2f}°",xy=(d['th'],dt(d['raw'])),xytext=(d['th']+dx,ytxt),fontsize=8,arrowprops=dict(arrowstyle='-',lw=.6))
ax.set_xlabel(r'Crystal angle $\theta$ (°)');ax.set_ylabel('Count rate (counts s$^{-1}$)');ax.set_title('X-ray spectrum of Cu anode, LiF crystal, $U_A$=35 kV, $I_A$=1 mA');ax.legend(loc='upper right')
save(fig,'lif_spectrum');plt.close()
fig,axs=plt.subplots(2,2,figsize=(9,6))
for a,(nm,n,lo,hi),d in zip(axs.flat,lif_pk,lif):
    p,e,r,chi=fit(L,lo,hi); k=(L[:,0]>=lo-.4)&(L[:,0]<=hi+.4)
    a.errorbar(L[k,0],dt(L[k,1]),sdt(L[k,1]),fmt='k.',ms=4,lw=.6,label='data')
    xx=np.linspace(lo,hi,300);a.plot(xx,g(xx,*p),'C3',label='Gaussian+linear fit')
    a.axvline(p[1],color='C0',ls='--',lw=.8)
    a.set_title(f"{nm} , n={n}:  $\\theta$ = {p[1]:.3f}$\\pm${e[1]:.3f}°",fontsize=9);a.set_xlabel(r'$\theta$ (°)');a.set_ylabel('rate (s$^{-1}$)')
axs[0,0].legend(fontsize=7)
fig.tight_layout();save(fig,'lif_fits');plt.close()
# ---------------- KBr -----------------
lamKa,lamKb=R['K$_\\alpha$']['lam'],R['K$_\\beta$']['lam']
kb_pk=[('K$_\\beta$',1,11.0,11.9),('K$_\\alpha$',1,12.2,13.4),('K$_\\beta$',2,23.7,24.7),('K$_\\alpha$',2,26.5,27.6),
       ('K$_\\beta$',3,38.0,39.0),('K$_\\alpha$',3,43.2,44.3),('K$_\\beta$',4,56.4,57.4),('K$_\\alpha$',4,67.8,69.0)]
kb=[]
for nm,n,lo,hi in kb_pk:
    p,e,r,chi=fit(K,lo,hi)
    lam=lamKa if 'alpha' in nm else lamKb
    th=p[1]
    kb.append(dict(name=nm,n=n,th=th,sth_fit=e[1],sth=(0.10 if e[1]>0.1 else float(np.hypot(e[1],0.05))),lam=lam,
        d_naive=n*lam/(2*np.sin(np.radians(th))),sinth=np.sin(np.radians(th))))
    print(kb[-1])
R['kb']=kb
th=np.array([d['th'] for d in kb]);sth=np.array([d['sth'] for d in kb]);nn=np.array([d['n'] for d in kb]);lam=np.array([d['lam'] for d in kb])
# model A: no offset, fit d only
fA=lambda X,d: np.degrees(np.arcsin(X[0]*X[1]/(2*d)))
pA,cA=curve_fit(fA,(nn,lam),th,p0=[330],sigma=sth,absolute_sigma=True)
chiA=np.sum(((th-fA((nn,lam),*pA))/sth)**2)/(len(th)-1)
# model B: offset
fB=lambda X,d,dl: np.degrees(np.arcsin(X[0]*X[1]/(2*d)))+dl
pB,cB=curve_fit(fB,(nn,lam),th,p0=[330,-0.8],sigma=sth,absolute_sigma=True)
resB=th-fB((nn,lam),*pB); chiB=np.sum((resB/sth)**2)/(len(th)-2)
sB=np.sqrt(np.diag(cB))*max(1,np.sqrt(chiB))
print('A',pA,np.sqrt(cA[0,0]),chiA); print('B',pB,sB,chiB,resB)
# propagate wavelength uncertainty (Ka,Kb common) -> d
def dfit(la,lb):
    l=np.where(['alpha' in d['name'] for d in kb],la,lb)
    p,_=curve_fit(fB,(nn,l),th,p0=[330,-0.8],sigma=sth);return p[0]
d0=dfit(lamKa,lamKb)
s_lam_d=np.hypot(dfit(lamKa+R['K$_\\alpha$']['slam'],lamKb+R['K$_\\beta$']['slam'])-d0,0)
print('d from lambda unc',s_lam_d)
R['kbfit']=dict(dA=pA[0],sdA=float(np.sqrt(cA[0,0])),chiA=chiA,dB=pB[0],sdB=sB[0],off=pB[1],soff=sB[1],chiB=chiB,s_lam_d=s_lam_d,res=list(resB))
# KBr figures
fig,ax=plt.subplots(figsize=(9,4.6))
ax.plot(K[:,0],K[:,1],color='0.6',lw=.9,label='Measured (uncorrected)');ax.plot(K[:,0],dt(K[:,1]),'C3',lw=1.1,label='Dead-time corrected')
for d in kb:
    yy=dt(K[np.argmin(abs(K[:,0]-d['th'])),1])
    ax.annotate(f"{d['name']}\nn={d['n']}",xy=(d['th'],yy),xytext=(d['th'],yy+60),ha='center',fontsize=7.5,arrowprops=dict(arrowstyle='-',lw=.5))
ax.set_ylim(0,850);ax.set_xlabel(r'Crystal angle $\theta$ (°)');ax.set_ylabel('Count rate (counts s$^{-1}$)');ax.set_title('X-ray spectrum of Cu anode, KBr crystal, $U_A$=35 kV, $I_A$=1 mA');ax.legend(loc='upper right')
save(fig,'kbr_spectrum');plt.close()
fig,axs=plt.subplots(1,2,figsize=(10,4.2))
ax=axs[0]
X=nn*lam/2   # = d sin(th-delta)
sinc=np.sin(np.radians(th-pB[1]))
ax.errorbar(X,np.sin(np.radians(th)),yerr=np.cos(np.radians(th))*np.radians(sth),fmt='o',color='C0',label=r'$\sin\theta_{\rm obs}$ (no correction)')
ax.errorbar(X,sinc,yerr=np.cos(np.radians(th))*np.radians(sth),fmt='s',color='C3',mfc='none',label=r'$\sin(\theta_{\rm obs}-\delta)$')
xx=np.linspace(0,X.max()*1.03,50);ax.plot(xx,xx/pB[0],'C3-',lw=1,label=f'Fit: slope $=1/d$,  $d$={pB[0]:.1f} pm')
ax.set_xlabel(r'$n\lambda/2$ (pm)');ax.set_ylabel(r'$\sin\theta$');ax.legend(fontsize=8);ax.set_title('KBr: Bragg plot')
ax=axs[1];ax.errorbar(nn+0.05*(np.array(['alpha' in d['name'] for d in kb])),resB,sth,fmt='ko');ax.axhline(0,color='r',lw=.8)
ax.set_xlabel('Order $n$');ax.set_ylabel('Residual $\\theta_{obs}-\\theta_{fit}$ (°)');ax.set_title('Residuals of the fit')
fig.tight_layout();save(fig,'kbr_fit');plt.close()
# ---------------- Absorption ----------------
files=[('no_absorber',0.00),('LiF_0.02mm',0.02),('lif_4mm',0.04),('0.06',0.06),('0.08',0.08),('0.1mm',0.10)]
A={d:load(f) for f,d in files}; ds=np.array([d for f,d in files])
ang=A[0.0][:,0]
for d in ds: assert np.allclose(A[d][:,0],ang)
fig,ax=plt.subplots(figsize=(9,4.6))
for d in ds: ax.plot(ang,dt(A[d][:,1]),lw=1.1,label=f'd = {d:.2f} mm')
ax.set_xlabel(r'Crystal angle $\theta$ (°)');ax.set_ylabel('Dead-time corrected rate (s$^{-1}$)');ax.set_title('Transmitted spectrum through Al absorbers (LiF, 5 mm aperture)');ax.legend(ncol=2)
save(fig,'abs_spectra');plt.close()
chosen=[15.0,16.0,17.0,18.0,19.0,20.2,21.0,23.5,24.0,25.0]
rho=2.70
R0=np.array([A[d][:,1] for d in ds])
def win(th,h):
    i=np.argmin(abs(ang-th)); return i,R0[:,max(0,i-h):i+h+1].mean(axis=1)
res=[]
fig,axs=plt.subplots(2,5,figsize=(14,5.6))
def wls(x,y,sy):
    w=1/sy**2; S=w.sum();Sx=(w*x).sum();Sy=(w*y).sum();Sxx=(w*x**2).sum();Sxy=(w*x*y).sum();D=S*Sxx-Sx**2
    b=(S*Sxy-Sx*Sy)/D; a0=(Sy-b*Sx)/S; sb=np.sqrt(S/D); chi=np.sum(w*(y-a0-b*x)**2)/(len(x)-2)
    return a0,b,sb*max(1,np.sqrt(chi)),chi
from scipy.optimize import curve_fit
for a,th0 in zip(axs.flat,chosen):
    h=0 if th0==20.2 else 2
    i,N0=win(th0,h); npt=min(i,h)+h+1
    N=dt(N0); sN=np.sqrt(N0/(T*npt))/(1-tau*N0)**2
    y=np.log(N); sy=sN/N
    a0,b,sb,chi=wls(ds,y,sy)
    mu=-b*10; smu=sb*10
    lam=2*dLiF*np.sin(np.radians(ang[i])); slam=lam/np.tan(np.radians(ang[i]))*np.radians(0.05)
    # method B: S exp(-mu d)+B
    try:
        pB_,cB_=curve_fit(lambda d,S,m,B:S*np.exp(-m*d)+B,ds,N,p0=[N[0],40,100],sigma=sN,absolute_sigma=True,bounds=([0,0,0],[1e5,500,1e4]),maxfev=20000)
        eB=np.sqrt(np.diag(cB_)); chiB=np.sum(((N-(pB_[0]*np.exp(-pB_[1]*ds)+pB_[2]))/sN)**2)/3
        muB=pB_[1]*10;smuB=eB[1]*10*max(1,np.sqrt(chiB));Bk=pB_[2];sBk=eB[2]
    except Exception: muB=smuB=Bk=sBk=np.nan
    res.append(dict(theta=ang[i],lam=lam,slam=slam,E=hc/(lam/1000)/1e3,mu=mu,smu=smu,mur=mu/rho,smur=smu/rho,chi=chi,N=list(N),sN=list(sN),raw=list(N0),
                    murB=muB/rho,smurB=smuB/rho,B=Bk,sB=sBk,lna=a0,tN0=tau*N0[0]))
    a.errorbar(ds,y,sy,fmt='ko',ms=3);xx=np.linspace(0,.105,10);a.plot(xx,a0+b*xx,'C3')
    a.set_title(f'θ={ang[i]:.1f}°, λ={lam:.1f} pm\nμ={mu:.1f} cm$^{{-1}}$',fontsize=8);a.set_xlabel('d (mm)');a.set_ylabel('ln N')
fig.tight_layout();save(fig,'abs_lnN');plt.close()
R['abs']=res
for r in res: print({k:(round(v,3) if not isinstance(v,list) else '') for k,v in r.items()})
def powfit(lam_nm,mr,smr):
    yl=np.log(mr);syl=smr/mr;X=np.log(lam_nm)
    lnC,n,sn,chi=wls(X,yl,syl)
    return n,sn,np.exp(lnC),chi
lam_nm=np.array([r['lam'] for r in res])/1000; mr=np.array([r['mur'] for r in res]); smr=np.array([r['smur'] for r in res]); slam_nm=np.array([r['slam'] for r in res])/1000
n1,sn1,C1,chi1=powfit(lam_nm,mr,smr)
kB=np.array([not np.isnan(r['murB']) and r['smurB']<0.6*r['murB'] and r['theta']>=18.9 and r['theta']!=20.2 for r in res])
mrB=np.array([r['murB'] for r in res]);smrB=np.array([r['smurB'] for r in res])
n2,sn2,C2,chi2_=powfit(lam_nm[kB],mrB[kB],smrB[kB])
print('Method A n',n1,sn1,C1,chi1,'| Method B n',n2,sn2,C2,chi2_,kB)
R['pl']=dict(n=n1,sn=sn1,C=C1,chi=chi1,nB=n2,snB=sn2,CB=C2,chiB=chi2_,kB=kB.tolist())
# NIST-based reference (log-log interpolation of 8,10,15 keV tabulated values for Al)
Eref=np.array([8.0,10.0,15.0]);Mref=np.array([50.33,26.23,7.955])
nref=lambda E: 50.33*(8.0/E)**2.92 if E<8 else float(np.exp(np.interp(np.log(E),np.log(Eref),np.log(Mref))))
for r in res: r['ref']=float(nref(r['E']))
# lambda^3 fit (weighted, through origin and with intercept) Method A
l3=lam_nm**3
wk=1/smr**2
Sw=wk.sum();Sx=(wk*l3).sum();Sy=(wk*mr).sum();Sxx=(wk*l3**2).sum();Sxy=(wk*l3*mr).sum();Dd=Sw*Sxx-Sx**2
slope=(Sw*Sxy-Sx*Sy)/Dd; icpt=(Sy-slope*Sx)/Sw; chil=np.sum(wk*(mr-icpt-slope*l3)**2)/(len(mr)-2)
ss=np.sqrt(Sw/Dd)*max(1,np.sqrt(chil)); si=np.sqrt(Sxx/Dd)*max(1,np.sqrt(chil))
R['pl'].update(slope=slope,sslope=ss,icpt=icpt,sicpt=si,chil=chil)
print('lam3',slope,ss,icpt,si,chil)
fig,axs=plt.subplots(1,3,figsize=(14,4.3))
ax=axs[0];ax.errorbar(lam_nm*1000,mr,smr,fmt='ko',ms=4,label='Method A (ln N fit)')
ax.errorbar(lam_nm[kB]*1000+0.8,mrB[kB],smrB[kB],fmt='s',color='C0',ms=4,mfc='none',label='Method B (with background)')
ax.plot(lam_nm*1000,[r['ref'] for r in res],'g^',ms=5,label='NIST (interpolated)')
xx=np.linspace(100,175,100);ax.plot(xx,C1*(xx/1000)**n1,'C3',lw=1,label=f'Method A power-law fit, n={n1:.2f}')
ax.set_xlabel('λ (pm)');ax.set_ylabel('μ/ρ (cm$^2$ g$^{-1}$)');ax.legend(fontsize=7);ax.set_title('Mass attenuation coefficient of Al')
ax=axs[1];ax.errorbar(lam_nm,mr,smr,fmt='ko',ms=4);ax.plot(xx/1000,C1*(xx/1000)**n1,'C3',label=f'fit: slope $n$ = {n1:.2f} ± {sn1:.2f}')
ax.errorbar(lam_nm[kB],mrB[kB],smrB[kB],fmt='s',color='C0',ms=4,mfc='none',label='Method B')
ax.set_xscale('log');ax.set_yscale('log');ax.set_xticks([0.11,0.13,0.15,0.17]);ax.set_xticklabels(['0.11','0.13','0.15','0.17']);ax.minorticks_off();ax.set_yticks([10,20,30,50,70]);ax.set_yticklabels(['10','20','30','50','70']);ax.set_xlabel('λ (nm)');ax.set_ylabel('μ/ρ (cm$^2$ g$^{-1}$)');ax.legend(fontsize=8);ax.set_title('Log–log plot');ax.grid(True,which='both',alpha=.3)
ax=axs[2];ax.errorbar(l3*1e3,mr,smr,fmt='ko',ms=4);xx3=np.linspace(0,l3.max()*1.05,10);ax.plot(xx3*1e3,icpt+slope*xx3,'C3',label='Method A: linear fit')
ax.errorbar(l3[kB]*1e3,mrB[kB],smrB[kB],fmt='s',color='C0',ms=4,mfc='none',label='Method B')
ax.set_xlabel('$\\lambda^3$ ($10^{-3}$ nm$^3$)');ax.set_ylabel('μ/ρ (cm$^2$ g$^{-1}$)');ax.legend(fontsize=8);ax.set_title('μ/ρ vs λ$^3$')
fig.tight_layout();save(fig,'abs_lambda');plt.close()


# ---------------- Bragg diagram ----------------
fig,ax=plt.subplots(figsize=(7,3.6));ax.axis('off');ax.set_aspect('equal')
for y in (0,-1.4): ax.plot([-0.8,6.6],[y,y],'k',lw=2)
ax.text(6.7,0,'plane 1',va='center');ax.text(6.7,-1.4,'plane 2',va='center')
ar=dict(arrowstyle='->',lw=2)
ax.annotate('',xy=(3,0),xytext=(0,1.732),arrowprops=dict(ar,color='b'));ax.annotate('',xy=(6,1.732),xytext=(3,0),arrowprops=dict(ar,color='b'))
ax.annotate('',xy=(2.2,-1.4),xytext=(-0.8,0.332),arrowprops=dict(ar,color='r'));ax.annotate('',xy=(5.2,0.332),xytext=(2.2,-1.4),arrowprops=dict(ar,color='r'))
ax.text(0.2,1.3,'incident',color='b');ax.text(5.0,1.4,'reflected',color='b')
ax.text(1.55,0.12,r'$\theta$');ax.annotate('',xy=(-0.3,-1.4),xytext=(-0.3,0),arrowprops=dict(arrowstyle='<->'));ax.text(-0.55,-0.75,'$d$')
ax.text(3,-2.1,r'Path difference $=2d\sin\theta=n\lambda$',ha='center')
ax.set_xlim(-1,8);ax.set_ylim(-2.4,2.1)
save(fig,'bragg_diagram');fig.savefig('figures/bragg_diagram.png',dpi=200);plt.close()
print('Done: figures written to ./figures')