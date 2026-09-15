"""Q42 annual adapter. Frozen solver kernels are imported, never edited."""
import sys, os, time, json, hashlib, csv, math, io, zipfile, copy, platform
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'C题'/'附件'/'附件1.xlsx').is_file())
PACKAGE=HERE.parent
PREREQ=PACKAGE/'shared'
sys.path.insert(0,str(PREREQ))
import runtime as old
import controllers as ctl
from runtime import View, Sources, read, sha, digest, encoded
from forecasting import array_hash

BRANCHES=('M0_P1','R2_P1','M0_P7')
LIMITS=dict(seconds=2400,lp=102000,fit=2250,bytes=1024**3)

def atomic_bytes(path,raw):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    for attempt in range(6):
        try: os.replace(tmp,path); break
        except PermissionError:
            if attempt==5: raise
            time.sleep(.02*2**attempt)
    return sha(path)

def save(path,obj): return atomic_bytes(path,encoded(obj))

def pack(path,obj):
    """Lossless float64 array arena plus JSON structure; no rounding/pickle."""
    arrays=[]; cursor=0
    def walk(x):
        nonlocal cursor
        if isinstance(x,np.ndarray):
            a=np.asarray(x,dtype=np.float64); ref={'__array__':[cursor,a.size,list(a.shape),x.dtype.str]}; cursor+=a.size; arrays.append(a.ravel()); return ref
        if isinstance(x,dict): return {k:walk(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)): return [walk(v) for v in x]
        return x
    tree=walk(obj); arena=np.concatenate(arrays) if arrays else np.array([],float)
    stream=io.BytesIO(); buf=io.BytesIO(); np.save(buf,arena,allow_pickle=False)
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        z.writestr('structure.json',encoded(tree)); z.writestr('arrays.npy',buf.getvalue())
    return atomic_bytes(path,stream.getvalue())

def unpack(path):
    with zipfile.ZipFile(path) as z:
        tree=json.loads(z.read('structure.json')); arena=np.load(io.BytesIO(z.read('arrays.npy')),allow_pickle=False)
    def walk(x):
        if isinstance(x,dict):
            if set(x)=={'__array__'}:
                at,n,shape,dtype=x['__array__']; return arena[at:at+n].reshape(shape).astype(dtype)
            return {k:walk(v) for k,v in x.items()}
        if isinstance(x,list): return [walk(v) for v in x]
        return x
    return walk(tree)

class Budget:
    """Durable prepaid batches: a crash never refunds potentially used calls."""
    def __init__(self,out,limits=LIMITS):
        self.out=Path(out); self.out.mkdir(parents=True,exist_ok=True); self.limits=limits
        self.path=self.out/'resources.json'; self.start=time.perf_counter(); self.timings={}; self.allowance={'lp':0,'fit':0}
        self.state=read(self.path) if self.path.exists() else dict(lp=0,fit=0,wall=0.,actual_lp=0,actual_fit=0)
        self.prior_wall=self.state['wall']; self.persist()
    def remaining(self): return self.limits['seconds']-self.prior_wall-(time.perf_counter()-self.start)
    def check(self):
        if self.remaining()<=0: raise RuntimeError('BUDGET_STOP: cumulative active time exhausted')
    def persist(self):
        self.state['wall']=self.prior_wall+time.perf_counter()-self.start; save(self.path,self.state)
    def reserve(self,kind,label):
        self.check(); kind='fit' if kind=='fit' else 'lp'
        if not self.allowance[kind]:
            size=min(16 if kind=='lp' and self.limits['lp']>100 else 1,self.limits[kind]-self.state[kind])
            if size<=0: raise RuntimeError('BUDGET_STOP: '+kind+' exhausted')
            self.state[kind]+=size; self.allowance[kind]=size; self.persist()
        self.allowance[kind]-=1; self.state['actual_'+kind]+=1
    def addtime(self,k,s): self.timings[k]=self.timings.get(k,0)+s
    def record(self):
        self.persist(); return dict(**self.state,timings=self.timings,limits=self.limits,reservation_note='unused durable reservations retained on resume')

class Bases(old.Bases):
    def get(self,view,day):
        assert view.mapping=='Q42' and day>=31 and view.position>=day*144
        key=('Q42',day)
        if key in self.hot: return [x.copy() for x in self.hot[key]]
        path=self.sources.stamp(ROOT/f'结果/Q2/R2/forecasts/{day:03d}.json')
        obj=read(path); manifest=read(self.sources.stamp(path.parent.parent/'manifest.json'))
        assert next(v for k,v in manifest['inputs'].items() if k.replace('\\','/').endswith('C题/附件/附件2.xlsx'))==self.sources.hashes[str(ROOT/'C题/附件/附件2.xlsx')]
        ks=(0,) if day==364 else (0,1)
        assert set(obj['predictions'])==set(map(str,ks))
        assert len(obj['refs'])==2*len(ks)
        for ref in obj['refs']:
            k=ref['k']; v=ref['series']; a=np.array(obj['predictions'][str(k)][v],float)
            assert k in ks and ref['publish_day']==day and ref['target_day']==day+k and ref['training_target_end_inclusive']<day
            assert a.shape==(144,) and ref['forecast_sha256']==hashlib.sha256(a.tobytes()).hexdigest()
            assert ref['forecast_version']=='received_R6_lambda1_window28_halflife14'
        vals=[np.concatenate([obj['predictions'][str(k)][v] for k in ks]) for v in ('load','pv')]
        assert all(np.isfinite(x).all() and min(x)>=0 for x in vals)
        meta=dict(day=day,mapping='Q42',path=str(path),hash=sha(path),refs=obj['refs'],histories={v:array_hash(view.history_before(day,v)) for v in ('load','pv')})
        self.hot[key]=vals; self.refs[key]=meta; return [x.copy() for x in vals]
    def risk(self,view,day):
        if not self.pool:
            root=ROOT/'结果/Q2/M0'
            with self.sources.stamp(root/'forecast_archive.csv').open(encoding='utf-8-sig',newline='') as f:
                for row in csv.DictReader(f):
                    d=int(row['publish_day_index']); t=int(row['slot'])-1
                    assert 21<=d<=364 and 0<=t<144
                    self.pool.setdefault(d,np.full(144,np.nan))
                    assert np.isnan(self.pool[d][t]); self.pool[d][t]=float(row['net_hat_kwh'])
                    assert abs((float(row['load_hat_kw'])-float(row['pv_hat_kw']))/6-self.pool[d][t])<1e-7
                    assert datetime.fromisoformat(row['date']).date()==(datetime(2025,1,1)+timedelta(days=d)).date()
            assert set(self.pool)==set(range(21,365)) and all(np.isfinite(x).all() for x in self.pool.values())
            with self.sources.stamp(root/'delta_pool.csv').open(encoding='utf-8-sig',newline='') as f:
                for row in list(csv.reader(f))[1:]: self.delta[int(row[0])]=np.array(row[2:],float)
            assert set(self.delta)==set(range(31,365))
        key=('risk',day)
        if key not in self.hot:
            past=(view.history_before(day,'load')-view.history_before(day,'pv'))/6
            errors=np.array([past[d]-self.pool[d] for d in range(21,day)])
            expected=np.array([np.quantile(errors[:,max(0,t-2):min(144,t+3)],.8,method='linear') for t in range(144)])
            err=float(max(abs(expected-self.delta[day]))); assert err<1e-6
            self.validation.append(dict(day=day,first_issue=21,last_issue=day-1,max_difference_kwh=err,rule='0.8 linear quantile, same slot +/-2, no wrap'))
            self.hot[key]=self.delta[day]
        return self.hot[key].copy()

class Prices(old.Prices):
    def __init__(self,budget):
        self.budget=budget; self.path=budget.out/'price_cache.json'; self.coeffs=read(self.path) if self.path.exists() else {}; self.records=[]; self.checked={}
        for c in self.coeffs.values(): self.validate_coefficient(c)
    def coefficient(self,view,day):
        key=(id(view),day)
        if key not in self.checked: self.checked[key]=super().coefficient(view,day)
        return self.checked[key]

class Q42(ctl.Q42):
    def __init__(self,view,budget,identity,bases,prices,method,model,energy=10800.,end=52560):
        position=view.position; view.position=4464
        super().__init__(view,budget,identity,bases,prices,method,model)
        view.position=position; self.position=position; self.energy=energy; self.end=end
    def midnight(self,day):
        self.load_base,self.pv_base=self.bases.get(self.access,day); n=(self.load_base-self.pv_base)/6
        count=144 if self.model=='M0' else len(n)
        p,meta=self.prices.predict(self.access,np.arange(self.position,self.position+count),n[:count]*6/1000,self.method,dict(base=digest(self.bases.refs[('Q42',day)]),midnight=day*144))
        if self.model=='M0':
            delta=self.bases.risk(self.access,day); demand=n[:144]+delta; cap=self.pv_base[:144]/6
            sol=ctl.m0.solve_plan_lp(demand,p,cap,self.energy,time_limit=min(10.,self.budget.remaining())); q=sol['q'] if sol['usable'] else ctl.m0.fallback_q(n[:144]); extra=dict(delta=delta,wcap=cap)
        else:
            demand=n; sol=ctl.r2.solve_midnight_lp(demand,p,self.energy,time_limit=min(10.,self.budget.remaining())); q=sol['Q'][:144] if sol['usable'] else ctl.r2.fallback_q(n[:144]); extra={}
        self.q0=ctl.normalize_orders(q,p[:144],self.corrections,self.position)
        self.plans.append(dict(position=self.position,stage='midnight',nbar=demand,energy=self.energy,price_record=meta,solution=sol,**extra))
        self.versions.append(dict(id=f'{self.model}_{self.method}_{day}',day=day,published_position=self.position,start=self.position,end=self.position+144,orders=self.q0.copy(),q0=self.q0.copy(),kind='original'))

def source_identity(sources):
    accepted=read(PREREQ/'work/final_acceptance.json'); assert accepted['status']=='PASS_ALL_REQUIRED_PRE_SOLVE_TESTS'
    for p,h in accepted['current_source_sha256'].items(): assert sha(p)==h, 'frozen code changed: '+p
    model=sources.stamp(PACKAGE/'02_完整数学模型_P1已确认.md')
    code={str(p.relative_to(ROOT)):sha(p) for p in HERE.glob('*.py')}
    code.update({str(p.relative_to(ROOT)):sha(p) for p in HERE.glob('*.ps1')})
    code.update({str(p.relative_to(ROOT)):sha(p) for p in HERE.glob('*.mjs')})
    return dict(question='Q42',branches=list(BRANCHES),model_sha256=sha(model),code=code,frozen_code=accepted['current_source_sha256'],inputs=dict(sources.hashes),mapping_id='Q42_endpoint_j_equals_i',timezone='Asia/Shanghai',delta_hours=1/6,units='kW/kWh/yuan_per_kWh',python=sys.version,numpy=np.__version__,prefix=sys.prefix,budget=LIMITS,start=4464,end=52560,P1=dict(W=1008,window=4032,min_samples=1008,epsilon=1e-6,rho_clip=[0,1]))

def january(data,sources):
    path=sources.stamp(ROOT/'结果/Q2/R2/january.npz')
    with np.load(path,allow_pickle=False) as z: a={k:z[k].copy() for k in ('q','c','s','z','w','E','e0','e_end')}
    n=(data['load_nodes'][:4464]-data['pv_nodes'][:4464]).reshape(31,144)/6; E=6000.; rows=[]
    for d in range(31):
        q=np.zeros(144) if d==0 else np.maximum(n[max(0,d-7):d].mean(axis=0),0)
        assert max(abs(q-a['q'][d]))<1e-6 and abs(a['e0'][d]-E)<1e-6
        for t in range(144):
            i=d*144+t; r=q[t]-n[d,t]
            c=0. if d==0 else min(max(r,0),5000/6,(10800-E)/.9)
            s=0. if d==0 else min(max(-r,0),5000/6,.9*(E-1200))
            g=max(-r-s,0); w=max(r-c,0); en=E+.9*c-s/.9
            assert max(abs(a[k][d,t]-v) for k,v in [('c',c),('s',s),('z',g),('w',w),('E',en)])<1e-6
            p=float(data['price_nodes'][i]); rows.append(dict(position=i,q0=float(q[t]),a=float(q[t]),c=c,s=s,g=g,w=w,E_before=E,E_after=en,ell=float(data['load_nodes'][i]/6),pv=float(data['pv_nodes'][i]/6),price=p,normal=p*q[t],adjustment=0.,emergency=5*p*g,total=p*(q[t]+5*g),published=d*144,version_id=f'January_{d}'))
            E=en
    fee=math.fsum(float(r['total']) for r in rows)
    assert abs(E-10800)<1e-6 and abs(fee-3563860.7506973064)<.01
    return dict(ledger=rows,final_energy=E,total=fee,source=str(path),sha256=sha(path),status='PASS_INDEPENDENT_JANUARY_REPLAY')

def new_controller(branch,data,budget,identity,bases,prices,position=4464,energy=10800.):
    model,method=branch.split('_'); v=View(data,'Q42','C2',position)
    return Q42(v,budget,identity,bases,prices,method,model,energy)

def bundle(c):
    return dict(name=c.model,method=c.method,start=c.ledger[0]['position'],ledger=c.ledger,plans=c.plans,versions=c.versions,corrections=c.corrections,access_log=c.access.log,base_ref=c.bases.refs[('Q42',c.ledger[0]['position']//144)])

def clear_day(c):
    c.ledger=[]; c.plans=[]; c.versions=[]; c.corrections=[]; c.access.log=[]; c.prices.records=[]
