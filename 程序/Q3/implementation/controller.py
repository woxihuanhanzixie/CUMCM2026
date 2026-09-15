"""Hourly planning, ten-minute native states and settlement; no annual entry."""
import copy, time
from datetime import datetime,timedelta
import numpy as np
from core import Problem,solve_lp,execute,LO,digest
from forecasting import base_predict,initial_base,fused_curve

class Controller:
    def __init__(self,access,budget,identity):
        self.access=access; self.budget=budget; self.identity=identity
        self.position=access.position; self.energy=6000.; self.q0=None; self.a=None; self.versions=[]; self.day_versions=[]
        self.load_base=None; self.pv_base=None; self.pv_curve=None; self.official_version=None; self.load_mean=None
        self.references=np.full(6,LO); self.plan_time=None; self.plans=[]; self.ledger=[]; self.forecasts=[]; self.fit_records=[]
        self.base_id=None; self.history_cutoff=0
    def forecast(self,day,slot):
        if slot==0:
            if day==0:
                bulletin=self.access.official_released_by(0,0); self.pv_base=initial_base(bulletin); self.pv_curve=self.pv_base.copy(); self.load_base=None
                self.official_version=(0,0); self.base_id=digest(self.pv_base); self.history_cutoff=0
                self.forecasts.append(dict(position=0,official_version=self.official_version,pv=self.pv_curve.copy(),load=None,history_cutoff=0))
                return
            histories={v:self.access.history_before(day,v) for v in ('load','pv')}; preds={'load':[],'pv':[]}
            for k in range(1 if day==364 else 2):
                for v in ('load','pv'):
                    a,r=base_predict(histories[v],day,k,v=='load',self.budget); preds[v].append(a); self.fit_records.append(r)
            self.load_base=np.concatenate(preds['load']); self.pv_base=np.concatenate(preds['pv']); self.history_cutoff=day*144
            self.base_id=digest(dict(day=day,cutoff=self.history_cutoff,source=self.identity['source_sha256'],histories=histories))
        if day==0 and slot in (36,72,108):
            self.load_mean=float(self.access.observations_completed_by(self.position,'load').mean()); self.load_base=np.full(288,self.load_mean); self.history_cutoff=self.position
        if self.access.policy!='C0' or slot==0:
            bulletin=self.access.official_released_by(day,slot); anchor=float(self.access.observations_completed_by(self.position,'pv')[-1])
            self.pv_curve=fused_curve(self.pv_base,slot,bulletin,anchor,jan1=day==0); self.official_version=(day,slot)
        self.forecasts.append(dict(position=self.position,official_version=self.official_version,pv=self.pv_curve.copy(),load=self.load_base.copy(),history_cutoff=self.history_cutoff,base_id=self.base_id))
    def publish(self,day,slot,orders,kind):
        if kind=='original':
            if slot!=0: raise ValueError('Original outside midnight')
            self.q0=np.asarray(orders).copy(); self.q0.setflags(write=False); self.a=self.q0.copy(); self.day_versions=[]
        else:
            if self.access.policy!='C2' or slot not in (36,72,108): raise ValueError('Unauthorized update')
            old=self.a[:slot].copy(); self.a[slot:]=orders
            if not np.array_equal(old,self.a[:slot]): raise AssertionError('Delivered contract changed')
        event=dict(id=f'{self.access.policy}_d{day}_s{slot}',policy=self.access.policy,day=day,slot=slot,published_position=self.position,kind=kind,start=day*144+slot,end=(day+1)*144,orders=np.asarray(orders).copy(),q0=self.q0.copy())
        self.versions.append(event); self.day_versions.append(event)
    def problem(self,day,slot,stage):
        count=min((day+2)*144,365*144)-self.position
        return Problem(self.load_base[slot:slot+count]/6,self.pv_curve[slot:slot+count]/6,np.tile(self.access.prices,2)[slot:slot+count],self.energy,144-slot,stage,None if stage=='original' else self.q0[slot:],None if stage=='original' else self.a[slot:])
    def step(self):
        i=self.position; day,slot=divmod(i,144)
        if i>=288 or self.access.position!=i: raise ValueError('Small entry limited to Jan1-Jan2')
        tick=time.perf_counter()
        if slot in (0,36,72,108): self.forecast(day,slot)
        self.budget.addtime('prediction_V4',time.perf_counter()-tick)
        if day==0 and slot==0: self.publish(0,0,np.zeros(144),'original')
        if slot%6==0 and i>=36:
            stage='original' if slot==0 else ('adjust' if slot in (36,72,108) and self.access.policy=='C2' else 'feedback')
            p=self.problem(day,slot,stage); result=solve_lp(p,self.budget,f'{self.access.policy}_i{i}_{stage}')
            if stage in ('original','adjust'): self.publish(day,slot,result['A'][:144-slot],stage)
            self.references=result['E'][:6].copy(); self.plan_time=i
            self.plans.append(dict(position=i,stage=stage,problem=p.record(),solution=result,official_version=self.official_version,history_cutoff=self.history_cutoff,base_id=self.base_id,internal_b_committed=False))
        tick=time.perf_counter(); A=float(self.a[slot]); q=float(self.q0[slot]); E=self.energy; price=float(self.access.prices[slot])
        self.access.commit(i,A); L,G=self.access.reveal_current_after_commit(i)
        R=float(self.references[slot%6]) if i>=36 else LO
        act=execute(E,A,L/6,G/6,R); self.energy=act['E']
        version=self.day_versions[-1]; start=datetime(2025,1,1)+timedelta(minutes=10*i)
        row=dict(policy=self.access.policy,position=i,day=day,slot=slot,delivery_start=start.isoformat(),delivery_end=(start+timedelta(minutes=10)).isoformat(),timezone='Asia/Shanghai',source_node_index=max(i-1,0),initial_missing_node_approximation=i==0,display_row_date=(start-timedelta(minutes=10)).date().isoformat(),q0=q,a_final=A,q0_created_at=day*144,final_version_id=version['id'],final_published_at=version['published_position'],order_owner_day=day,ell=L/6,pv=G/6,price=price,E_before=E,E_after=self.energy,R=R,hour_plan_time=self.plan_time,official_version=self.official_version,base_id=self.base_id,history_cutoff=self.history_cutoff,**{k:act[k] for k in ('c','s','g','w')})
        row.update(normal=price*A,adjustment=.5*price*abs(A-q),emergency=5*price*act['g']); row['total']=row['normal']+row['adjustment']+row['emergency']
        self.ledger.append(row); self.access.complete(); self.position+=1
        self.budget.addtime('execution_V4',time.perf_counter()-tick); self.budget.check(); return row
    def snapshot(self):
        keys=('position','energy','q0','a','versions','day_versions','load_base','pv_base','pv_curve','official_version','load_mean','references','plan_time','plans','ledger','forecasts','fit_records','base_id','history_cutoff')
        return copy.deepcopy(dict(identity=self.identity,policy=self.access.policy,state={k:getattr(self,k) for k in keys},access_log=self.access.log))
    def restore(self,record):
        if record['identity']!=self.identity or record['policy']!=self.access.policy or record['state']['position']!=self.access.position: raise ValueError('Checkpoint source/code/protocol/policy identity mismatch')
        for k,v in record['state'].items(): setattr(self,k,copy.deepcopy(v))
        for k in ('q0','a','load_base','pv_base','pv_curve','references'):
            if getattr(self,k) is not None: setattr(self,k,np.asarray(getattr(self,k),dtype=float))
        if self.q0 is not None: self.q0.setflags(write=False)
        self.official_version=tuple(self.official_version); self.access.log=copy.deepcopy(record['access_log'])
