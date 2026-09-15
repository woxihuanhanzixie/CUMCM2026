"""Bounded Q4 adapters for accepted policies. Only Jan1-2 / Feb1-2 are executable."""
import copy,time
import numpy as np
import scipy.optimize
from runtime import View,Prices,Bases,digest,write,read
from core import Problem,solve_lp,execute
from controller import Controller
from forecasting import fused_curve
import m0_kernel as m0
import r2_kernel as r2
from greedy import greedy_step

def hook_q2(budget):
    def call(*a,**kw):
        budget.reserve('lp','Q2 primary_or_secondary'); options=kw.setdefault('options',{}); options['time_limit']=min(options.get('time_limit',10.),budget.remaining()); options['threads']=1
        t=time.perf_counter(); out=scipy.optimize.linprog(*a,**kw); budget.addtime('Q2_LP',time.perf_counter()-t); return out
    m0.linprog=call; r2.linprog=call
def normalize_orders(q,phat,events,r):
    q=np.asarray(q,float).copy(); bad=q<0
    if np.any(q < -1e-9): raise ValueError('negative order exceeds original numerical-zero policy')
    if bad.any():
        bound=float(1.5*np.max(phat)*(-q[bad]).sum())
        if bound>1e-6: raise ValueError('predicted normalization fee bound exceeded')
        events.append(dict(r=r,kind='order_zero',indices=np.flatnonzero(bad),raw=q[bad].copy(),forecast_fee_bound=bound)); q[bad]=0
    return q
class Q3(Controller):
    def __init__(self,view,budget,identity,bases,prices,method,start):
        super().__init__(view,budget,identity)
        if start not in (0,4464): raise ValueError('only approved two-day fixtures')
        self.bases=bases; self.prices=prices; self.method=method; self.start=start; self.end=start+288; self.last_price=None; self.corrections=[]
    def forecast(self,day,slot):
        if day==0: return super().forecast(day,slot)
        if slot==0:
            self.load_base,self.pv_base=self.bases.get(self.access,day); self.base_id=digest(self.bases.refs[(self.access.mapping,day)]); self.history_cutoff=day*144
        if self.access.policy!='C0' or slot==0:
            official=self.access.official_released_by(day,slot); anchor=float(self.access.observations_completed_by(self.position,'pv')[-1]); self.pv_curve=fused_curve(self.pv_base,slot,official,anchor); self.official_version=(day,slot)
        self.forecasts.append(dict(position=self.position,official_version=self.official_version,pv=self.pv_curve.copy(),load=self.load_base.copy(),history_cutoff=self.history_cutoff,base_id=self.base_id))
    def problem(self,day,slot,stage):
        count=min((day+2)*144,52560)-self.position; ell=self.load_base[slot:slot+count]/6; pv=self.pv_curve[slot:slot+count]/6
        phat,meta=self.prices.predict(self.access,np.arange(self.position,self.position+count),(ell-pv)*6/1000,self.method,dict(base_id=self.base_id,official_version=self.official_version))
        self.last_price=meta
        return Problem(ell,pv,phat,self.energy,144-slot,stage,None if stage=='original' else self.q0[slot:],None if stage=='original' else self.a[slot:])
    def publish(self,day,slot,orders,kind):
        prices=np.ones(len(orders)) if self.last_price is None else self.last_price['prices'][:len(orders)]
        q=normalize_orders(orders,prices,self.corrections,self.position)
        return super().publish(day,slot,q,kind)
    def step(self):
        i=self.position; day,slot=divmod(i,144)
        if not self.start<=i<self.end or self.access.position!=i: raise ValueError('outside two-day test domain')
        if slot in (0,36,72,108): self.forecast(day,slot)
        if i==0: self.publish(0,0,np.zeros(144),'original')
        if slot%6==0 and i>=36:
            stage='original' if slot==0 else ('adjust' if slot in (36,72,108) and self.access.policy=='C2' else 'feedback')
            p=self.problem(day,slot,stage); result=solve_lp(p,self.budget,f'{self.access.policy}_{self.method}_{i}_{stage}')
            if stage in ('original','adjust'): self.publish(day,slot,result['A'][:144-slot],stage)
            self.references=result['E'][:6].copy(); self.plan_time=i
            self.plans.append(dict(position=i,stage=stage,problem=p.record(),solution=result,price_record=self.last_price,official_version=self.official_version,history_cutoff=self.history_cutoff,base_id=self.base_id))
        a=float(self.a[slot]); q=float(self.q0[slot]); e=self.energy
        self.access.commit(i,a); load,pv,price=self.access.delivered(); R=float(self.references[slot%6]) if i>=36 else 1200.
        act=execute(e,a,load/6,pv/6,R); self.energy=act['E']; version=self.day_versions[-1]
        row=dict(position=i,policy=self.access.policy,method=self.method,mapping='Q43',q0=q,a=a,ell=load/6,pv=pv/6,price=price,E_before=e,E_after=self.energy,R=R,plan_time=self.plan_time,version_id=version['id'],published=version['published_position'],**{k:act[k] for k in ('c','s','g','w')})
        row.update(normal=price*a,adjustment=.5*price*abs(a-q),emergency=5*price*act['g']); row['total']=row['normal']+row['adjustment']+row['emergency']
        self.ledger.append(row); self.access.complete(); self.position+=1; self.budget.check(); return row
    def checkpoint(self):
        return dict(base=super().snapshot(),method=self.method,start=self.start,end=self.end,last_price=self.last_price,corrections=self.corrections)
    def load_checkpoint(self,obj):
        if obj['method']!=self.method or obj['start']!=self.start or obj['end']!=self.end: raise ValueError('checkpoint policy/domain mismatch')
        super().restore(obj['base']); self.last_price=copy.deepcopy(obj['last_price']); self.corrections=copy.deepcopy(obj['corrections'])
class Q42:
    def __init__(self,view,budget,identity,bases,prices,method,model):
        if model not in ('M0','R2') or (model=='R2' and method!='P1') or view.position!=4464: raise ValueError('only approved Q42 fixtures')
        self.access=view; self.budget=budget; self.identity=identity; self.bases=bases; self.prices=prices; self.method=method; self.model=model; self.start=4464; self.end=4752; self.position=view.position; self.energy=6000.
        self.q0=None; self.load_base=None; self.pv_base=None; self.ledger=[]; self.plans=[]; self.versions=[]; self.corrections=[]
    def midnight(self,day):
        self.load_base,self.pv_base=self.bases.get(self.access,day); n=(self.load_base-self.pv_base)/6
        count=144 if self.model=='M0' else 288
        p,meta=self.prices.predict(self.access,np.arange(self.position,self.position+count),n[:count]*6/1000,self.method,dict(base=digest(self.bases.refs[('Q42',day)]),midnight=day*144))
        if self.model=='M0':
            delta=self.bases.risk(self.access,day); demand=n[:144]+delta; cap=self.pv_base[:144]/6; sol=m0.solve_plan_lp(demand,p,cap,self.energy,time_limit=min(10.,self.budget.remaining())); q=sol['q'] if sol['usable'] else m0.fallback_q(n[:144]); extra=dict(delta=delta,wcap=cap)
        else:
            demand=n; sol=r2.solve_midnight_lp(demand,p,self.energy,time_limit=min(10.,self.budget.remaining())); q=sol['Q'][:144] if sol['usable'] else r2.fallback_q(n[:144]); extra={}
        self.q0=normalize_orders(q,p[:144],self.corrections,self.position)
        self.plans.append(dict(position=self.position,stage='midnight',nbar=demand,energy=self.energy,price_record=meta,solution=sol,**extra))
        self.versions.append(dict(id=f'{self.model}_{self.method}_{day}',day=day,published_position=self.position,start=self.position,end=self.position+144,orders=self.q0.copy(),q0=self.q0.copy(),kind='original'))
    def step(self):
        i=self.position; day,t=divmod(i,144)
        if not self.start<=i<self.end or self.access.position!=i: raise ValueError('outside Q42 fixture')
        if t==0: self.midnight(day)
        q=float(self.q0[t]); e=self.energy; self.access.commit(i,q)
        if self.model=='R2':
            nbar=(self.load_base[t:]-self.pv_base[t:])/6; count=len(nbar)
            p,meta=self.prices.predict(self.access,np.arange(i,i+count),nbar*6/1000,self.method,dict(midnight=day*144))
            nbar=nbar.copy(); nbar[0]=self.access.root_net()
            sol=r2.solve_intraday_lp(nbar,p,e,self.q0[t:],time_limit=min(10.,self.budget.remaining())); restored=None
            if sol['usable']:
                restored=r2.restore_solution(nbar,p,self.q0[t:],*(sol[k] for k in ('Q','c','s','z','w','E')))
                if not all(restored[k] for k in ('feas_ok','cs_ok','zw_ok','czF_ok','obj_not_increased')): raise ValueError('R2 restoration failed')
                act=dict(c=float(restored['c'][0]),s=float(restored['s'][0]),g=float(restored['z'][0]),w=float(restored['w'][0]),E=float(restored['E'][0]))
            else:
                raw=greedy_step(e,float(nbar[0]),q); act=dict(c=raw['c'],s=raw['s'],g=raw['z'],w=raw['w'],E=raw['e'])
            self.plans.append(dict(position=i,stage='root',nbar=nbar,energy=e,q_fixed=self.q0[t:].copy(),price_record=meta,solution=sol,restored=restored))
        load,pv,price=self.access.delivered()
        if self.model=='M0':
            raw=greedy_step(e,(load-pv)/6,q); act=dict(c=raw['c'],s=raw['s'],g=raw['z'],w=raw['w'],E=raw['e'])
        self.energy=act['E']; version=self.versions[-1]
        row=dict(position=i,mapping='Q42',policy=self.model,method=self.method,q0=q,a=q,ell=load/6,pv=pv/6,price=price,E_before=e,E_after=self.energy,R=None,plan_time=i if self.model=='R2' else day*144,version_id=version['id'],published=version['published_position'],**{k:act[k] for k in ('c','s','g','w')})
        row.update(normal=price*q,adjustment=0.,emergency=5*price*act['g']); row['total']=row['normal']+row['emergency']; self.ledger.append(row); self.access.complete(); self.position+=1; self.budget.check(); return row
    def checkpoint(self):
        names=('position','energy','q0','load_base','pv_base','ledger','plans','versions','corrections')
        return copy.deepcopy(dict(identity=self.identity,method=self.method,model=self.model,state={k:getattr(self,k) for k in names},access_log=self.access.log))
    def load_checkpoint(self,obj):
        if obj['identity']!=self.identity or obj['method']!=self.method or obj['model']!=self.model or obj['state']['position']!=self.access.position: raise ValueError('checkpoint mismatch')
        for k,v in obj['state'].items(): setattr(self,k,copy.deepcopy(v))
        for k in ('q0','load_base','pv_base'): setattr(self,k,np.asarray(getattr(self,k),float))
        self.access.log=copy.deepcopy(obj['access_log'])
def make(branch,data,budget,identity,bases,prices,position=None):
    name,method,start=branch; start=int(start)
    view=View(data,'Q42' if name in ('M0','R2') else 'Q43',name if name.startswith('C') else 'C2',start)
    obj=Q42(view,budget,identity,bases,prices,method,name) if name in ('M0','R2') else Q3(view,budget,identity,bases,prices,method,start)
    if position is not None: obj.access.position=position
    return obj
