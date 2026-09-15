"""E1-LP10 primary LP; independent D1 MILP is only a bounded test reference."""
from dataclasses import dataclass, asdict
import time, json, hashlib, warnings, os
from pathlib import Path
import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix

P=5000/6; ETA=.9; LO=1200.; HI=10800.
def serial(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,Path): return str(x)
    raise TypeError(type(x).__name__)
def save(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=serial),encoding='utf-8')
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,default=serial).encode()).hexdigest()

class Budget:
    def __init__(self,out,start=None):
        self.out=Path(out); self.start=time.perf_counter() if start is None else start
        self.deadline=float(os.environ.get('Q3_VALIDATION_DEADLINE_EPOCH',time.time()+1200))
        self.calls=0; self.fits=0; self.events=[]; self.timings={}
    def remaining(self): return min(1200-(time.perf_counter()-self.start),self.deadline-time.time())
    def check(self):
        if self.remaining()<=0: raise RuntimeError('20 minute validation deadline exhausted')
    def reserve(self,kind,label):
        self.check()
        if kind=='fit':
            if self.fits>=12: raise RuntimeError('12 fit budget exhausted')
            self.fits+=1
        else:
            if self.calls>=200: raise RuntimeError('200 optimization budget exhausted')
            self.calls+=1
        self.events.append(dict(kind=kind,label=label,elapsed=time.perf_counter()-self.start))
        # Native DLL crashes bypass Python exception handling: persist BEFORE work.
        save(self.out/'resource_progress.json',self.record())
    def addtime(self,key,seconds): self.timings[key]=self.timings.get(key,0)+seconds
    def record(self): return dict(optimizer_calls=self.calls,fits=self.fits,wall_seconds=time.perf_counter()-self.start,remaining_deadline_seconds=self.remaining(),timings=self.timings,events=self.events)

@dataclass
class Problem:
    ell:object; pv:object; prices:object; energy:float; today:int; stage:str; q0:object=None; current:object=None
    def __post_init__(self):
        for k in ('ell','pv','prices'): setattr(self,k,np.asarray(getattr(self,k),dtype=float).copy())
        n=len(self.ell)
        if n<1 or any(x.shape!=(n,) or not np.isfinite(x).all() for x in (self.ell,self.pv,self.prices)): raise ValueError('Invalid vectors')
        if min(self.ell.min(),self.pv.min())<0 or self.prices.min()<=0: raise ValueError('Positive prices and nonnegative power required')
        if not LO-1e-7<=self.energy<=HI+1e-7 or not 1<=self.today<=n or self.stage not in ('original','adjust','feedback'): raise ValueError('Invalid state/stage')
        for k in ('q0','current'):
            x=getattr(self,k); x=np.zeros(self.today) if x is None else np.asarray(x,dtype=float).copy()
            if x.shape!=(self.today,) or not np.isfinite(x).all() or (x<0).any(): raise ValueError('Invalid contract')
            setattr(self,k,x)
    def record(self): return asdict(self)

class Matrix:
    def __init__(self,n): self.n=n; self.r=[]; self.c=[]; self.v=[]; self.lo=[]; self.hi=[]
    def add(self,terms,lo,hi):
        row=len(self.lo); self.lo.append(lo); self.hi.append(hi)
        for c,v in terms.items(): self.r.append(row); self.c.append(c); self.v.append(v)
    def matrix(self): return coo_matrix((self.v,(self.r,self.c)),shape=(len(self.lo),self.n)).tocsr()

def solve_lp(p,budget,label):
    n=len(p.ell); cost=np.zeros(4*n); bounds=[]; eq=Matrix(4*n); ub=Matrix(4*n); const=0.
    for t in range(n):
        j=4*t; net=p.ell[t]-p.pv[t]; price=p.prices[t]; fixed=p.stage=='feedback' and t<p.today
        adj=t<p.today and p.stage!='original'; q=p.q0[t] if t<p.today else 0.
        U=p.ell[t]+P
        if adj: U=max(U,p.q0[t],p.current[t])
        if fixed:
            A=p.current[t]; r=A-net
            low,high=(0.,ETA*min(P,r)) if r>=0 else (-min(P,-r)/ETA,0.)
            ab=(A,A)
        else: low,high=-min(P,max(net,0))/ETA,ETA*P; ab=(0,U)
        bounds.extend([(low,high),(LO,HI),ab,(0,U+q) if adj else (0,0)])
        terms={j+1:1,j:-1}
        if t: terms[j-3]=-1
        eq.add(terms,p.energy if t==0 else 0,p.energy if t==0 else 0)
        cost[j+2]=price
        if adj:
            cost[j+3]=.5*price
            ub.add({j+2:1,j+3:-1},-np.inf,q); ub.add({j+2:-1,j+3:-1},-np.inf,-q)
        if not fixed:
            ub.add({j:ETA,j+2:-1},-np.inf,-net); ub.add({j:1/ETA,j+2:-1},-np.inf,-net)
        elif r<0: cost[j]=5*price*ETA; const+=5*price*(-r)
    ae=eq.matrix(); au=ub.matrix(); be=np.array(eq.hi); bu=np.array(ub.hi); bd=np.array(bounds)
    budget.reserve('lp',label); tick=time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result=linprog(cost,A_ub=au,b_ub=bu,A_eq=ae,b_eq=be,bounds=bounds,method='highs-ds',options={'time_limit':min(5.,budget.remaining()),'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'threads':1,'parallel':False,'random_seed':20260913})
    sec=time.perf_counter()-tick; budget.addtime('LP',sec)
    if not result.success:
        save(budget.out/'failure_problem.json',dict(label=label,problem=p.record(),message=result.message)); raise RuntimeError(result.message)
    x=result.x; resid=max(np.max(abs(ae@x-be)),np.max(np.maximum(au@x-bu,0),initial=0),np.max(np.maximum(bd[:,0]-x,0)),np.max(np.maximum(x-bd[:,1],0)))
    dual=float(be@result.eqlin.marginals+bu@result.ineqlin.marginals+bd[:,0]@result.lower.marginals+bd[:,1]@result.upper.marginals+const)
    stationarity=float(np.max(abs(cost-ae.T@result.eqlin.marginals-au.T@result.ineqlin.marginals-result.lower.marginals-result.upper.marginals)))
    A=x[2::4].copy(); u=x[::4]; E=x[1::4].copy(); c=np.maximum(u,0)/ETA; s=np.maximum(-u,0)*ETA
    gap=p.ell-p.pv-A+c-s; g=np.maximum(gap,0); w=np.maximum(-gap,0)
    kappa=np.zeros(n); kappa[:p.today]=.5 if p.stage!='original' else 0.
    q=np.r_[p.q0,np.zeros(n-p.today)]; fee=p.prices*(A+kappa*abs(A-q)+5*g)
    upper=float(fee.sum()); obj=float(result.fun+const)
    if resid>1e-6 or stationarity>1e-7 or abs(upper-obj)>1e-6 or abs(upper-dual)>1e-5: raise AssertionError(('LP certificate',label,resid,stationarity,upper-obj,upper-dual))
    for a,b in ((c,s),(c,g),(s,w),(g,w)):
        if np.any((a>1e-6)&(b>1e-6)): raise AssertionError('LP physical mode')
    budget.check()
    return dict(A=A,u=u,E=E,c=c,s=s,g=g,w=w,upper=upper,lower=dual,objective=obj,residual=float(resid),stationarity=stationarity,seconds=sec,input_hash=digest(p.record()))

def reference_milp(p,budget,label):
    # Independent full physical formulation, no candidate matrix or elimination.
    n=len(p.ell); nv=8*n; mat=Matrix(nv); cost=np.zeros(nv); low=np.zeros(nv); high=np.zeros(nv); integer=np.zeros(nv)
    for t in range(n):
        a,c,s,g,w,e,d,m=range(t*8,t*8+8); locked=t<p.today and p.stage=='feedback'; adj=t<p.today and p.stage!='original'
        q=p.q0[t] if t<p.today else 0.; U=p.ell[t]+P
        if adj: U=max(U,p.q0[t],p.current[t])
        if locked: U=p.current[t]; low[a]=U
        high[a]=U; high[c]=high[s]=P; high[g]=p.ell[t]; high[w]=U+p.pv[t]; low[e]=LO; high[e]=HI; high[d]=U+q if adj else 0; high[m]=1; integer[m]=1
        cost[a]=p.prices[t]; cost[g]=5*p.prices[t]; cost[d]=.5*p.prices[t] if adj else 0
        mat.add({a:1,s:1,g:1,c:-1,w:-1},p.ell[t]-p.pv[t],p.ell[t]-p.pv[t])
        terms={e:1,c:-ETA,s:1/ETA}
        if t: terms[e-8]=-1
        mat.add(terms,p.energy if t==0 else 0,p.energy if t==0 else 0)
        mat.add({c:1,m:-P},-np.inf,0); mat.add({s:1,m:P},-np.inf,P)
        mat.add({g:1,m:p.ell[t]},-np.inf,p.ell[t]); mat.add({w:1,m:-(U+p.pv[t])},-np.inf,0)
        if adj:
            mat.add({a:1,d:-1},-np.inf,q); mat.add({a:-1,d:-1},-np.inf,-q)
    matrix=mat.matrix(); budget.reserve('milp_reference',label); tick=time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r=milp(cost,integrality=integer,bounds=Bounds(low,high),constraints=LinearConstraint(matrix,mat.lo,mat.hi),options={'time_limit':min(5.,budget.remaining()),'mip_rel_gap':0.,'mip_abs_gap':1e-9,'mip_feasibility_tolerance':1e-9,'threads':1,'random_seed':20260913})
    sec=time.perf_counter()-tick; budget.addtime('MILP_reference',sec)
    if not r.success or r.mip_dual_bound is None:
        save(budget.out/'failure_reference.json',dict(problem=p.record(),message=r.message)); raise RuntimeError('Uncertified reference '+r.message)
    ax=matrix@r.x
    resid=max(np.max(np.maximum(np.array(mat.lo)-ax,0)),np.max(np.maximum(ax-np.array(mat.hi),0)),np.max(np.maximum(low-r.x,0)),np.max(np.maximum(r.x-high,0)),np.max(abs(r.x[7::8]-np.rint(r.x[7::8]))))
    if resid>1e-6 or r.fun-r.mip_dual_bound>1e-6: raise AssertionError(('Reference certificate',resid,r.fun-r.mip_dual_bound))
    budget.check(); return dict(upper=float(r.fun),lower=float(r.mip_dual_bound),residual=float(resid),seconds=sec,x=r.x)

def execute(E,A,ell,pv,R):
    if not all(np.isfinite([E,A,ell,pv,R])) or not LO-1e-7<=E<=HI+1e-7 or not LO-1e-7<=R<=HI+1e-7 or min(A,ell,pv)<-1e-7: raise ValueError('Invalid execution input')
    r=A+pv-ell
    if r>=0: c=min(r,P,max(0,(HI-E)/ETA)); s=g=0.; w=r-c
    else: c=w=0.; s=min(-r,P,ETA*max(E-R,0)); g=-r-s
    return dict(c=c,s=s,g=g,w=w,E=E+ETA*c-s/ETA)
