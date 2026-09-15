"""History-only forecast service and observation-only rolling controller."""
from __future__ import annotations
import runtime  # noqa: F401
from pathlib import Path
import hashlib
import json
import numpy as np
from model import build_tree, fit_params, solve
from v2_impl.common.forecast import RidgeForecaster
from v2_impl.common.environment import greedy_step


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,
                                    separators=(',',':')).encode()).hexdigest()


def forecast_at(history_load, history_pv, dates, publish):
    if len(history_load)!=publish or len(history_pv)!=publish:
        raise ValueError('forecast interface requires exactly the past days')
    L=np.full((len(dates),144),np.nan);G=L.copy()
    L[:publish]=history_load;G[:publish]=history_pv
    fc=RidgeForecaster(dict(load_act=L,pv_act=G,dates=dates))
    result=[]
    for k in range(min(2,len(dates)-publish)):
        pr=fc.predict_day(publish,k)
        arr=np.stack([pr['load'],pr['pv'],(pr['load']-pr['pv'])/6])
        if not np.isfinite(arr).all():
            raise ValueError('forecast touched unavailable data or produced nonfinite values')
        result.append(arr)
    return np.stack(result)


class ForecastArchive:
    def __init__(self, folder, dates):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.dates=dates

    def get(self,h,history_load,history_pv):
        if len(history_load)!=h or len(history_pv)!=h:
            raise ValueError('history boundary mismatch')
        path=self.folder/f'{h:03d}.npz'
        cutoff=hashlib.sha256(np.asarray(history_load).tobytes()+np.asarray(history_pv).tobytes()).hexdigest()
        if path.exists():
            with np.load(path,allow_pickle=False) as f:
                if str(f['history_hash'])!=cutoff:
                    raise ValueError('forecast archive history hash mismatch')
                return f['forecast'].copy()
        forecast=forecast_at(history_load,history_pv,self.dates,h)
        temporary=path.with_suffix('.tmp')
        with temporary.open('wb') as f:
            np.savez_compressed(f,forecast=forecast,history_hash=cutoff,publish=h,history_end=h-1)
        temporary.replace(path)
        return forecast

    def snapshot(self,day,history_load,history_pv):
        if len(history_load)!=day or len(history_pv)!=day:
            raise ValueError('snapshot receives past days only')
        blocks=[]
        for h in range(max(21,day-29),day-1):
            pr=self.get(h,history_load[:h],history_pv[:h])
            actual=(history_load[h:h+2]-history_pv[h:h+2])/6
            blocks.append((h,(actual-pr[:,2]).reshape(-1)))
        params=fit_params(blocks,day)
        pr=self.get(day,history_load,history_pv)
        return dict(day=day,base=pr[:,2].reshape(-1).tolist(),params=params,
                    history_max_day=day-1,forecast_hash=hashlib.sha256(pr.tobytes()).hexdigest())


class Controller:
    def __init__(self,price,time_limit=5.,gap=1e-4,formulation='reduced'):
        self.price=np.tile(price,2)
        self.options=dict(time_limit=time_limit,gap=gap,formulation=formulation)

    def optimize(self,snapshot,energy,q=None,slot=0,observed=None,ct4=False):
        tree=build_tree(snapshot['base'],snapshot['params'],start=slot,observed=observed,ct4=ct4)
        fixed=None if q is None else {(('today',),u):float(q[u]) for u in range(slot,144)}
        res=solve(tree['nodes'],self.price,energy,fixed=fixed,**self.options)
        res.log.update(day=snapshot['day'],slot=slot,kind='midnight' if observed is None else 'intraday',
                       energy_before=energy,observed_net=observed,branches=tree['branches'],
                       leaves=len(tree['leaves']),snapshot_hash=digest(snapshot),
                       input_hash=digest(dict(snapshot=snapshot,energy=energy,
                                             q=None if q is None else np.asarray(q).tolist(),
                                             slot=slot,observed=observed,ct4=ct4)),
                       observed_max_slot=None if observed is None else slot)
        return res,tree

    def midnight(self,snapshot,energy):
        res,tree=self.optimize(snapshot,energy)
        if res.usable:
            q=np.array([res.orders[(('today',),u)] for u in range(144)])
        else:
            q=np.maximum(snapshot['base'][:144],0)
        res.log['fallback']=not res.usable
        return q,res.log

    def step(self,snapshot,energy,q,slot,observed):
        # No future actuals are accepted by this interface.
        res,tree=self.optimize(snapshot,energy,q,slot,observed)
        if res.usable:
            if tree['nodes'][0]['slot']!=slot or sum(n['slot']==slot for n in tree['nodes'])!=1:
                raise ValueError('current measured action must be unique')
            action={k:float(res.actions[k][0]) for k in ('c','s','z','w','E')}
        else:
            action=greedy_step(energy,observed,float(q[slot]));action['E']=action.pop('e')
        res.log['fallback']=not res.usable
        return action,res.log
