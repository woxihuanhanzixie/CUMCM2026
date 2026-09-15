"""Annual adapter. step is copied from verified Controller with only domain guard changed."""
import time, copy
from datetime import datetime,timedelta
import numpy as np
from core import solve_lp as verified_solve_lp,execute,LO
from runtime import atomic
from controller import Controller
from forecasting import fused_curve

def solve_lp(problem,budget,label):
    budget.write(budget.out/'current_problem.json',dict(label=label,problem=problem.record()))
    try: return verified_solve_lp(problem,budget,label)
    except BaseException as exc:
        atomic(budget.out/'failure_problem.json',dict(label=label,problem=problem.record(),error=repr(exc)))
        raise

class AnnualController(Controller):
    def __init__(self,access,budget,identity,cache,end_position=52560):
        super().__init__(access,budget,identity)
        if not 0<end_position<=52560: raise ValueError('Invalid annual end')
        self.cache=cache; self.end_position=end_position
    def publish(self,day,slot,orders,kind):
        orders=np.asarray(orders,dtype=float).copy()
        negative=orders<0
        if np.any(orders < -1e-9): raise ValueError('Negative contract exceeds numerical zero tolerance')
        if negative.any():
            indices=np.flatnonzero(negative); correction=-orders[negative]
            fee_bound=float(1.5*np.max(self.access.prices)*correction.sum())
            if fee_bound>1e-6: raise ValueError('Contract zero normalization exceeds fee tolerance')
            self.access._event('contract_numerical_zero',slots=(indices+slot).tolist(),raw_values=orders[negative].tolist(),max_fee_change=fee_bound)
            orders[negative]=0.
        return super().publish(day,slot,orders,kind)
    def forecast(self,day,slot):
        if day==0: return super().forecast(day,slot)
        if slot==0:
            obj=self.cache.get(self.access,day)
            self.load_base=np.array(obj['predictions']['load']); self.pv_base=np.array(obj['predictions']['pv'])
            self.base_id=obj['key']; self.history_cutoff=day*144
            self.fit_records=obj['tasks']
        if self.access.policy!='C0' or slot==0:
            bulletin=self.access.official_released_by(day,slot)
            anchor=float(self.access.observations_completed_by(self.position,'pv')[-1])
            self.pv_curve=fused_curve(self.pv_base,slot,bulletin,anchor)
            self.official_version=(day,slot)
        self.forecasts.append(dict(position=self.position,official_version=self.official_version,pv=self.pv_curve.copy(),load=self.load_base.copy(),history_cutoff=self.history_cutoff,base_id=self.base_id))
    def clear_day(self):
        self.plans=[]; self.ledger=[]; self.versions=[]; self.forecasts=[]; self.fit_records=[]; self.access.log=[]
    def state(self):
        keys=('position','energy','q0','a','day_versions','load_base','pv_base','pv_curve','official_version','load_mean','references','plan_time','base_id','history_cutoff')
        return copy.deepcopy(dict(identity=self.identity,policy=self.access.policy,state={k:getattr(self,k) for k in keys},access_log=[]))
    def block(self):
        return dict(policy=self.access.policy,ledger=self.ledger,plans=self.plans,versions=self.versions,forecasts=self.forecasts,access_log=self.access.log,fit_records=self.fit_records)
    def step(self):
        i=self.position; day,slot=divmod(i,144)
        if i>=self.end_position or self.access.position!=i: raise ValueError('Outside configured annual domain')
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
