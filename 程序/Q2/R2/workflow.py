"""Causal input adapter, resumable driver, version identity and compact evidence."""
import runtime
from runtime import ROOT
import csv, hashlib, json, sys, time, platform
from pathlib import Path
import numpy as np
import scipy
from v2_impl import m1
from v2_impl.common.data_io import load_attachment1,load_attachment2,A1_DEFAULT,A2_DEFAULT,TEMPLATE_DEFAULT,sha256_file
from v2_impl.common.forecast import RidgeForecaster,fit_series,predict_series,training_window
from v2_impl.common.january import run_january,check_january

def plain(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,Path): return str(x)
    raise TypeError(type(x).__name__)

def save(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=plain,allow_nan=False),encoding='utf-8')
    tmp.replace(path)

def digest_array(a): return hashlib.sha256(np.asarray(a,dtype='<f8').tobytes()).hexdigest()

def identity():
    sources=[ROOT/'workflow.py',ROOT/'runtime.py',ROOT/'run_m1.py',ROOT/'config.json']+sorted((ROOT/'v2_impl').rglob('*.py'))
    info={'source':{str(p.relative_to(ROOT)):sha256_file(p) for p in sources},
          'inputs':{str(p):sha256_file(p) for p in (A1_DEFAULT,A2_DEFAULT,TEMPLATE_DEFAULT)},
          'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
          'model':'M1_D48','revision':'R2_20260912'}
    info['signature']=hashlib.sha256(json.dumps(info,sort_keys=True).encode()).hexdigest()
    return info

def load_data():
    a1=load_attachment1();a2=load_attachment2()
    for a in (a1['price'],a2['load_act'],a2['pv_act']):
        if not np.isfinite(a).all(): raise ValueError('nonfinite original input')
    if (a2['load_act']<0).any() or (a2['pv_act']<0).any(): raise ValueError('negative original power')
    return {**a2,'price':a1['price'],'labels':a1['labels']}

class PublishedForecast(RidgeForecaster):
    """Copy ONLY history before d. Actual current/future cells are never supplied.
    Same received feature/fit/predict arithmetic. Fits retained for reproducibility.
    """
    def __init__(self,data,d):
        bounded={'dates':data['dates']}
        for key in ('load_act','pv_act'):
            bounded[key]=np.full((365,144),np.nan)
            bounded[key][:d]=data[key][:d]
        super().__init__(bounded,horizons=(0,1))
        self.publish=d; self.cache={};self.fits={};self.refs=[]
    def predict_day(self,d,k):
        if d!=self.publish or k not in (0,1) or d+k>=365: raise ValueError('forecast information boundary')
        if k not in self.cache:
            values={};lo,hi=training_window(d,k)
            for name,is_load in (('load',True),('pv',False)):
                y=self.load_act if is_load else self.pv_act
                fit=fit_series(y,d,k,self.dates,is_load)
                values[name]=predict_series(y,d,k,self.dates,is_load,fit) if fit is not None else self._predict_series(y,d,k,is_load,None)
                if not np.isfinite(values[name]).all(): raise ValueError('forecast read unavailable actuals')
                self.fits[f'{name}_k{k}']=fit
                self.refs.append({'series':name,'k':k,'publish_day':d,'publish_date':self.dates[d].isoformat(),
                    'target_day':d+k,'target_date':self.dates[d+k].isoformat(),
                    'history_cutoff':self.dates[d-1].strftime('%Y-%m-%d')+'T24:00:00',
                    'training_publish_start':lo,'training_publish_end_inclusive':hi-1,
                    'training_target_start':lo+k,'training_target_end_inclusive':hi-1+k,
                    'n_train':self.n_train(d,k),'forecast_version':'received_R6_lambda1_window28_halflife14',
                    'method':'ridge' if fit is not None else 'fallback_S', 'forecast_sha256':digest_array(values[name])})
            self.cache[k]=values
        return self.cache[k]

def planner_for(data,d):
    fc=PublishedForecast(data,d)
    return m1.M1Planner({'price':data['price'],'dates':data['dates']},fc)

def audit_day(rec,data):
    """Independent actual ledger; does not use common.ledger or LP restore helper."""
    d=rec['day'];q,c,s,z,w,E=(np.asarray(rec[k],float) for k in ('q','c','s','z','w','E'))
    if any(a.shape!=(144,) or not np.isfinite(a).all() for a in (q,c,s,z,w,E)): raise ValueError('damaged trajectory')
    n=(data['load_act'][d]-data['pv_act'][d])/6
    previous=np.r_[rec['e0'],E[:-1]]
    balance=float(np.max(np.abs(q-c+s+z-w-n)))
    state=float(np.max(np.abs(E-previous-.9*c+s/.9)))
    physical=max(0.,float(-min(q.min(),c.min(),s.min(),z.min(),w.min())),float(1200-E.min()),float(E.max()-10800),float(max(c.max(),s.max())-5000/6),float(np.minimum(c,s).max()),float(np.minimum(c,z).max()),float(np.minimum(z,w).max()))
    plan=float(sum(float(p)*float(v) for p,v in zip(data['price'],q)))
    emergency=float(sum(5*float(p)*float(v) for p,v in zip(data['price'],z)))
    cost_diff=max(abs(plan-rec['plan_cost']),abs(emergency-rec['emg_cost']),abs(plan+emergency-rec['total_cost']))
    lock=digest_array(q)==rec['lock_sha256']
    return {'passed':max(balance,state,physical)<=1e-6 and cost_diff<=.01 and lock,
            'balance_kwh':balance,'state_kwh':state,'physical_kwh':physical,'bill_difference_yuan':cost_diff,
            'locked':lock,'plan_cost':plan,'emergency_cost':emergency,'total_cost':plan+emergency,'end_energy':float(E[-1])}

def compact_lp(r,store,prefix):
    out={}
    for k,v in r.items():
        if isinstance(v,np.ndarray): store[prefix+k]=v
        elif k in ('raw_lp','restored_lp','primary_lp'):
            for name,arr in v.items():store[prefix+k+'_'+name]=arr
        else:out[k]=v
    return out

def execute(days=2,resume=False,wall_s=300,annual_authorized=False):
    if days not in range(1,335) or wall_s<=0: raise ValueError('invalid run budget')
    if days>2 and not annual_authorized: raise ValueError('Annual run requires explicit user authorization')
    if days<=2 and wall_s>300: raise ValueError('two-day wall budget cannot exceed 300 seconds')
    start=time.perf_counter();deadline=start+wall_s
    out=Path(__import__('os').environ.get('CUMCM_RUN_OUT',str(ROOT/'outputs/M1_D48')));out.mkdir(parents=True,exist_ok=True)
    manifest=identity();manifest_path=out/'manifest.json'
    if manifest_path.exists():
        if not resume: raise ValueError('existing output: use resume; never overwrite run')
        if json.loads(manifest_path.read_text(encoding='utf-8'))['signature']!=manifest['signature']: raise ValueError('signature mismatch; cannot reuse checkpoint')
    else:save(manifest_path,manifest)
    data=load_data()
    jan=run_january(data['price'],data['load_act'][:31],data['pv_act'][:31])
    if check_january(jan):raise ValueError(check_january(jan))
    np.savez_compressed(out/'january.npz',**jan)
    save(out/'january_summary.json',{k:jan[k] for k in ('plan_cost','emg_cost','jan_cost','feb1_energy')})
    existing=sorted((out/'days').glob('*.json')) if (out/'days').exists() else []
    recs=[json.loads(p.read_text(encoding='utf-8')) for p in existing]
    for i,r in enumerate(recs):
        if r['day']!=31+i:raise ValueError('checkpoint date gap')
        if abs(r['e0']-(jan['feb1_energy'] if i==0 else recs[i-1]['e_end']))>1e-6 or not audit_day(r,data)['passed']:raise ValueError('checkpoint state/ledger corrupted')
        npz_path=out/'lp'/f"{r['day']:03d}.npz"
        if sha256_file(npz_path)!=r['lp_archive_sha256']:raise ValueError('LP checkpoint damaged')
    e=float(recs[-1]['e_end'] if recs else jan['feb1_energy'])
    calls=0; paused=False; error=None
    try:
        for d in range(31+len(recs),31+days):
            if time.perf_counter()+10>=deadline:paused=True;break
            planner=planner_for(data,d);partial=out/'partial.json';archive={}
            if partial.exists():
                rec=json.loads(partial.read_text(encoding='utf-8'))
                if rec['day']!=d or rec['e0']!=e or rec['lock_sha256']!=digest_array(rec['q']):raise ValueError('invalid partial checkpoint')
                with np.load(out/'partial.npz') as f:archive={k:f[k] for k in f.files}
                # Recreate only causal forecast cache, no extra midnight solve.
                planner.horizon_arrays(d)
                e_cur=rec['E'][-1] if rec['E'] else e
            else:
                midnight=planner.plan_midnight(d,e);calls+=1
                q=midnight['q'].copy();q.flags.writeable=False
                rec={'day':d,'date':data['dates'][d].isoformat(),'e0':e,'q':q.tolist(),'lock_sha256':digest_array(q),
                    'midnight':compact_lp(midnight,archive,'midnight_'),'intraday':[],**{k:[] for k in ('c','s','z','w','E')}}
                # Persist lock before any current actual is revealed.
                save(out/'locks'/f'{d:03d}.json',{'day':d,'q':q,'e0':e,'sha256':rec['lock_sha256'],'event':'locked_before_first_observation'})
                e_cur=e
            save(out/'forecasts'/f'{d:03d}.json',{'refs':planner.fc.refs,'fits':planner.fc.fits,'predictions':planner.fc.cache})
            for t in range(len(rec['E']),144):
                if time.perf_counter()+10>=deadline or (days<=2 and calls+2>600):paused=True;break
                current=float((data['load_act'][d,t]-data['pv_act'][d,t])/6)
                r=planner.intraday_lp(d,t,e_cur,np.asarray(rec['q']),current_actual=current);calls+=r['solver_calls']
                for k in ('c','s','z','w'):rec[k].append(r[k])
                rec['E'].append(r['e']);e_cur=r['e']
                rec['intraday'].append(compact_lp(r,archive,f't{t:03d}_'))
            if paused:
                save(partial,rec);np.savez_compressed(out/'partial.npz',**archive);break
            rec['e_end']=e_cur;rec['plan_cost']=float(np.dot(data['price'],rec['q']))
            rec['emg_cost']=float(5*np.dot(data['price'],rec['z']));rec['total_cost']=rec['plan_cost']+rec['emg_cost']
            rec['audit']=audit_day(rec,data)
            if not rec['audit']['passed']:raise RuntimeError('actual ledger failed')
            (out/'lp').mkdir(exist_ok=True);lp_path=out/'lp'/f'{d:03d}.npz'
            np.savez_compressed(lp_path,**archive);rec['lp_archive_sha256']=sha256_file(lp_path)
            save(out/'days'/f'{d:03d}.json',rec)
            if partial.exists():partial.unlink()
            recs.append(rec);e=e_cur
            print(json.dumps({'day':d,'total':rec['total_cost'],'E_end':e,'wall_s':time.perf_counter()-start}),flush=True)
    except Exception as ex:
        error=f'{type(ex).__name__}: {ex}'
        if 'rec' in locals():save(out/'failure_partial.json',rec)
        if 'archive' in locals():np.savez_compressed(out/'failure_partial.npz',**archive)
        save(out/'failure.json',{'error':error,'wall_s':time.perf_counter()-start})
        raise
    finally:
        elapsed=time.perf_counter()-start
        inv={'wall_s':elapsed,'new_lp_calls':calls,'requested_days':days,'paused':paused,'error':error}
        with (out/'invocations.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(inv)+'\n')
        summarize(out,recs,jan,inv)
    return inv

def summarize(out,recs,jan,inv):
    logs=[]
    for r in recs:
        logs.append({'day':r['day'],'kind':'midnight','slot':-1,**{k:r['midnight'].get(v) for k,v in [('status','lp_status'),('fallback','fallback'),('wall_s','wall_s')]}})
        logs.extend({'day':r['day'],'kind':'intraday',**{k:x[k] for k in ('slot','status','fallback','wall_s')}} for x in r['intraday'])
    fb=[x for x in logs if x['fallback']]
    intraday=[x for r in recs for x in r['intraday']]
    summary={'model':'M1_D48','revision':'R2_20260912','period':([recs[0]['date'],recs[-1]['date']] if recs else []),
        'complete_days':len(recs),'slots':len(recs)*144,'decision_count':len(logs),
        'lp_calls':len(recs)+sum(x['solver_calls'] for x in intraday),
        'secondary_attempted':sum(x['secondary']['attempted'] for x in intraday),
        'secondary_accepted':sum(x['secondary']['accepted'] for x in intraday),
        'secondary_fallback_count':sum(not x['secondary']['accepted'] for x in intraday if not x['fallback']),
        'plan_cost':sum(r['plan_cost'] for r in recs),'emergency_cost':sum(r['emg_cost'] for r in recs),
        'total_cost':sum(r['total_cost'] for r in recs),'end_energy':recs[-1]['e_end'] if recs else jan['feb1_energy'],
        'january_cost':jan['jan_cost'],'annual_complete':len(recs)==334,'two_day_verified':len(recs)>=2 and all(r['audit']['passed'] for r in recs[:2]),
        'independent_audit_passed':False,'independent_audit_scope':'pending separate replay','user_adopted':False,
        'solver_status_counts':{str(s):sum(x['status']==s for x in logs) for s in sorted(set(x['status'] for x in logs))},
        'feasible_solutions':sum(not x['fallback'] for x in logs),'fallback_count':len(fb),'invocation':inv}
    save(out/'summary.json',summary)
    with (out/'solver_log.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=['day','kind','slot','status','fallback','wall_s']);wr.writeheader();wr.writerows(logs)
    with (out/'trajectory.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.writer(f);wr.writerow(['day_index','date','slot','q_kwh','c_kwh','s_kwh','z_kwh','w_kwh','E_kwh'])
        for r in recs:
            for t in range(144):wr.writerow([r['day'],r['date'],t+1]+[r[k][t] for k in ('q','c','s','z','w','E')])
    for name,fields,rows in (
        ('daily_summary.csv',['day','date','e0','e_end','plan_cost','emg_cost','total_cost'],recs),
        ('forecast_refs.csv',None,[ref for p in sorted((out/'forecasts').glob('*.json')) for ref in json.loads(p.read_text(encoding='utf-8'))['refs']])):
        if not rows:continue
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            wr=csv.DictWriter(f,fieldnames=fields or list(rows[0]),extrasaction='ignore');wr.writeheader();wr.writerows(rows)
    save(out/'fallback_log.json',[{'day':r['day'],'e0':r['e0'],'midnight':r['midnight'] if r['midnight']['fallback'] else None,
        'intraday':[x for x in r['intraday'] if x['fallback']],'plan_cost':r['plan_cost'],'emg_cost':r['emg_cost']}
        for r in recs if r['midnight']['fallback'] or any(x['fallback'] for x in r['intraday'])])
    with (out/'midnight_orders.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.writer(f);wr.writerow(['day','date']+list(range(1,145)))
        for r in recs:wr.writerow([r['day'],r['date']]+r['q'])
    months={}
    for r in recs:
        b=months.setdefault(r['date'][:7],{'plan_cost':0.,'emg_cost':0.,'total_cost':0.})
        for k in b:b[k]+=r[k]
    save(out/'monthly_summary.json',months)
