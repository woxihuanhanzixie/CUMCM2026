"""Causal R6/PV ridge and forward-hold official fusion. No source file access."""
from datetime import date,timedelta
import hashlib
import numpy as np

def array_hash(a): return hashlib.sha256(np.asarray(a,dtype='<f8').tobytes()).hexdigest()
def training_days(d,k): return list(range(max(14,d-28-k),d-k))

def feature_rows(history,h,k,load):
    if h<14 or h>len(history) or k not in (0,1): raise ValueError('Unavailable feature history')
    m3=history[h-3:h].mean(axis=0)
    phase=np.arange(144)*2*np.pi/144
    columns=[history[h-1],history[h+k-7],history[h+k-14],m3,history[h-7:h].mean(axis=0),m3-history[h-6:h-3].mean(axis=0),np.full(144,history[h-1].mean())]
    for j in (1,2,3): columns.extend([np.sin(j*phase),np.cos(j*phase)])
    if load:
        weekday=(date(2025,1,1)+timedelta(days=h+k)).weekday()
        columns.extend([np.full(144,float(weekday==j)) for j in range(1,7)])
    return np.column_stack(columns)

def base_predict(history,d,k,load,budget=None):
    history=np.asarray(history,dtype=float)
    if history.shape!=(d,144) or d<1 or k not in (0,1) or d+k>=365: raise ValueError('Invalid causal history/target domain')
    if not np.isfinite(history).all() or (history<0).any(): raise ValueError('Invalid history; no imputation')
    hs=training_days(d,k)
    record=dict(d=d,k=k,load=load,history_hash=array_hash(history),training_days=hs,target_days=[h+k for h in hs],feature_count=19 if load else 13,method='fallback',max_completed_day=d-1)
    if len(hs)<7:
        src=d+k-7 if load and 0<=d+k-7<d else d-1
        raw=history[src].copy(); record.update(fallback_day=src,raw=raw)
        return raw,record
    if budget is not None: budget.reserve('fit',f'base_d{d}_k{k}_{"load" if load else "pv"}')
    X=np.vstack([feature_rows(history,h,k,load) for h in hs]); y=np.concatenate([history[h+k] for h in hs])
    daily_w=2.**(-(d-1-np.array(hs)-k)/14); weights=np.repeat(daily_w,144)
    mean=np.average(X,axis=0,weights=weights); std=np.sqrt(np.average((X-mean)**2,axis=0,weights=weights))
    scale=np.where(std<1e-12 if load else std<=1e-12,1.,std)
    Z=(X-mean)/scale; intercept=float(np.average(y,weights=weights))
    design=np.vstack([Z*np.sqrt(weights[:,None]),np.eye(Z.shape[1])])
    target=np.r_[(y-intercept)*np.sqrt(weights),np.zeros(Z.shape[1])]
    beta=np.linalg.lstsq(design,target,rcond=None)[0]
    augmented=np.column_stack([np.ones(len(Z)),Z]); penalty=np.diag([0.]+[1.]*Z.shape[1])
    lhs=augmented.T@(weights[:,None]*augmented)+penalty; rhs=augmented.T@(weights*y)
    theta=np.r_[intercept,beta]; residual=float(np.max(abs(lhs@theta-rhs))/max(1.,np.max(abs(rhs))))
    if residual>1e-7: raise ValueError('Ridge stationarity failed')
    raw=intercept+((feature_rows(history,d,k,load)-mean)/scale)@beta
    record.update(method='ridge',mean=mean,scale=scale,intercept=intercept,beta=beta,weights=daily_w,raw=raw,stationarity=residual,design_hash=array_hash(X),target_hash=array_hash(y))
    return np.maximum(raw,0),record

def initial_base(official):
    official=np.asarray(official,dtype=float)
    if official.shape!=(24,) or not np.isfinite(official).all() or (official<0).any(): raise ValueError('Invalid first bulletin')
    starts=np.arange(144)/6
    one=np.interp(starts,np.arange(25),np.r_[official[0],official])
    return np.tile(one,2)

def fused_curve(base,release_slot,official,anchor,jan1=False):
    base=np.asarray(base,dtype=float); official=np.asarray(official,dtype=float)
    if official.shape!=(24,) or not np.isfinite(official).all() or (official<0).any() or not np.isfinite(base).all() or (base<0).any(): raise ValueError('Invalid fusion data')
    if release_slot not in (0,36,72,108): raise ValueError('Illegal release time')
    if not np.isfinite(anchor) or anchor<0: raise ValueError('Missing legal completed anchor')
    nodes=np.r_[anchor,official]; result=base.copy()
    for m in range(release_slot,min(len(base),release_slot+145)):
        lead=m-release_slot
        if lead%6==0:
            result[m]=nodes[lead//6]; continue
        h=lead//6; theta=(lead%6)/6
        if jan1: value=(1-theta)*nodes[h]+theta*nodes[h+1]
        else:
            left=release_slot+6*h; right=left+6
            # A right node outside the prediction grid occurs only at the end
            # of the natural-day domain. Its midnight baseline is unavailable;
            # build a boundary value explicitly in the caller (see base_boundary).
            bleft=base[left]
            bright=base[right] if right<len(base) else base_boundary(base,right)
            value=base[m]+(1-theta)*(nodes[h]-bleft)+theta*(nodes[h+1]-bright)
        result[m]=max(0.,value)
    return result

def base_boundary(base,index):
    # Forward-hold predictions on a finite domain have a left limit at end.
    # This is used only as an interpolation baseline node at the exclusive end,
    # never as an additional predicted/controlled segment or 2026 horizon.
    if index!=len(base): raise ValueError('Baseline target beyond domain')
    return float(base[-1])
