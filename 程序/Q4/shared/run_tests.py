"""Required pre-solve validation, bounded to eight two-day fixtures. No annual mode."""
import os,sys,time,json,copy,traceback,warnings
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
sys.dont_write_bytecode=True
from runtime import *
from controllers import make,hook_q2
from independent import raw_sources,audit,export_boundaries
from forecasting import base_predict
BRANCHES=[('C0','P1',0),('C1','P1',0),('C2','P1',0),('M0','P1',4464),('M0','P7',4464),('R2','P1',4464),('C2','P1',4464),('C2','P7',4464)]
def name(branch): return '_'.join(map(str,branch))
def reject(fn):
    try: fn()
    except (ValueError,AssertionError): return
    raise AssertionError('bad identity/source accepted')
def freeze(sources):
    for rec in read(HERE/'vendor_provenance.json'):
        assert sha(rec['source'])==rec.get('source_sha256',rec['sha256']) and sha(rec['target'])==rec['sha256']
        sources.stamp(rec['source'])
    for p in list(HERE.glob('*.py'))+list((HERE/'vendor').glob('*.py'))+[HERE/'vendor_provenance.json']+list(HERE.parent.glob('0[012]_*.md')): sources.stamp(p)
    return json.loads(encoded(dict(model='Q4-P1-prereq-v1',sources=dict(sources.hashes),mathematics='02 confirmed P1',branches=BRANCHES,only_small_tests=True)))
def save_checkpoint(path,obj):
    value=obj.checkpoint(); write(path,dict(payload=value,sha256=digest(value))); loaded=read(path); assert loaded['sha256']==digest(loaded['payload']); return loaded
def restore_checked(obj,env):
    if digest(env['payload'])!=env['sha256']: raise ValueError('checkpoint content checksum failed')
    obj.load_checkpoint(env['payload'])
def bundle(obj,branch):
    return dict(name=branch[0],method=branch[1],start=branch[2],ledger=obj.ledger,plans=obj.plans,versions=obj.versions,forecasts=getattr(obj,'forecasts',[]),corrections=obj.corrections,access_log=obj.access.log)
def compare_decision(a,b):
    # Timings are not deterministic. Compare all selected numerical decision and price arrays.
    x=a.plans[-1]; y=b.plans[-1]; errors=[]
    assert x['position']==y['position'] and x['stage']==y['stage']
    errors.append(float(np.max(abs(np.array(x['price_record']['prices'])-y['price_record']['prices']))))
    for k in ('A','Q','q','E','c','s','z','g','w'):
        if k in x['solution'] and k in y['solution']: errors.append(float(np.max(abs(np.array(x['solution'][k])-y['solution'][k]))))
    assert max(errors,default=0.)<1e-6; return max(errors,default=0.)
def prepare(data,sources,budget,bases,prices,identity):
    checks=[]
    for mapping in ('Q42','Q43'):
        for day in (31,32):
            v=View(data,mapping,position=day*144); base=bases.get(v,day)
            if mapping=='Q42': bases.risk(v,day)
            if day==31:
                for idx,var in enumerate(('load','pv')):
                    # Independent augmented-LS computation also checks the old Q42 normal-equation cache.
                    y,record=base_predict(v.history_before(day,var),day,0,var=='load',budget)
                    err=float(max(abs(y-base[idx][:144]))); assert err<1e-6
                    checks.append(dict(mapping=mapping,day=day,series=var,max_error=err))
        v=View(data,mapping,position=144); b=bases.get(v,1); assert all(x.shape==(288,) for x in b)
    sources.verify()
    report=dict(status='PASS',base_recalculation=checks,risk_cache=bases.validation,base_sources={str(k):v for k,v in bases.refs.items()},source_sha256=sources.hashes)
    write(STATE/'prepared.json',report); return report
def pilot(data,sources,budget,bases,prices,identity):
    records=[]
    for branch,steps in [(('R2','P1',4464),6),(('M0','P1',4464),6),(('C2','P1',0),42),(('C2','P1',4464),6)]:
        obj=make(branch,data,budget,identity,bases,prices); tick=time.perf_counter(); before=budget.calls
        for _ in range(steps): obj.step()
        elapsed=time.perf_counter()-tick
        records.append(dict(branch=name(branch),steps=steps,calls=budget.calls-before,seconds=elapsed))
        write(STATE/'pilot'/f'{name(branch)}.json.gz',bundle(obj,branch))
    r2=records[0]; q3=records[-1]; m0=records[1]
    # Conservative 2x numerical pace + 30 seconds input/audit/persistence, no CPU-core division.
    estimate=2*(r2['seconds']/r2['calls']*590+q3['seconds']/q3['calls']*250+m0['seconds']*8)+30
    report=dict(status='PASS_WIRING_PILOT',records=records,full_test_estimate_seconds=estimate,auto_run_allowed=estimate<120,notes='No annual runtime claim. Estimate covers remaining fixtures with audit/I-O allowance.')
    write(STATE/'pilot_report.json',report); return report
def full(data,sources,budget,bases,prices,identity):
    raw=raw_sources(); reports={}; bundles=[]; recovery=[]; causal=[]
    for branch in BRANCHES:
        obj=make(branch,data,budget,identity,bases,prices); folder=STATE/'full'/name(branch); checkpoint36=None; tick=time.perf_counter(); bc=budget.calls
        while obj.position<obj.end:
            if obj.position==obj.start+36: checkpoint36=save_checkpoint(folder/'before_6h.json.gz',obj)
            if obj.position==obj.start+143:
                env=save_checkpoint(folder/'before_midnight.json.gz',obj)
                obj.step(); obj.step(); original=obj
                resumed=make(branch,data,budget,identity,bases,prices,position=branch[2]+143); restore_checked(resumed,env)
                bad=copy.deepcopy(env); bad['payload']['method']='INVALID'; reject(lambda:restore_checked(resumed,bad))
                wrong=make(branch,data,budget,dict(identity,invalid=True),bases,prices,position=branch[2]+143); reject(lambda:restore_checked(wrong,env))
                resumed.step(); resumed.step()
                err=compare_decision(original,resumed)
                assert abs(original.energy-resumed.energy)<1e-6 and digest(original.ledger)==digest(resumed.ledger) and digest(original.versions)==digest(resumed.versions)
                recovery.append(dict(branch=name(branch),checkpoint_position=branch[2]+143,replayed_segments=2,max_decision_error=err,exact_ledger=True,corrupt_checkpoint_rejected=True,wrong_identity_rejected=True)); obj=resumed
            else: obj.step()
        block=bundle(obj,branch); path=folder/'result.json.gz'; write(path,block); loaded=read(path); assert digest(loaded)==digest(block)
        report=audit(loaded,raw); report.update(seconds=time.perf_counter()-tick,calls=budget.calls-bc); reports[name(branch)]=report; bundles.append(loaded)
        # Causal complete-decision checks. Mutated inputs are test warehouses only.
        if branch[0] in ('M0',):
            r=branch[2]; variant=copy.deepcopy(data)
            for k in ('load_nodes','pv_nodes','price_nodes'): variant[k][r:]*=1.37
            mutated=make(branch,variant,budget,identity,bases,prices); mutated.step()
            reference=make(branch,data,budget,identity,bases,prices); reference.plans=[obj.plans[0]]
            err=compare_decision(reference,mutated)
        else:
            r=branch[2]+36; j=r if branch[0]=='R2' else r-1; variant=copy.deepcopy(data)
            variant['price_nodes'][j:]*=1.37
            # R2 root actual net is authorized after commitment; preserve it, disturb later actuals.
            jload=j+1 if branch[0]=='R2' else j
            variant['load_nodes'][jload:]+=5000.; variant['pv_nodes'][jload:]*=.3
            variant['official']={k:(v+50000 if k[0]*144+k[1]>r or (branch[0]=='C0' and k[1]!=0) else v.copy()) for k,v in variant['official'].items()}
            mutated=make(branch,variant,budget,identity,bases,prices,position=r); restore_checked(mutated,checkpoint36); mutated.step()
            reference=make(branch,data,budget,identity,bases,prices); reference.plans=[p for p in obj.plans if p['position']==r][-1:]
            err=compare_decision(reference,mutated)
        causal.append(dict(branch=name(branch),position=r,max_decision_error=err,root_current_net_preserved=branch[0]=='R2',future_actuals_and_unauthorized_bulletins_changed=True))
        print(json.dumps(dict(branch=name(branch),status='PASS',seconds=report['seconds'],cost=report['total'],calls=report['calls']),ensure_ascii=False),flush=True)
    # Check forecasts using legally generated curves against actuals; fixed scoring windows only.
    metrics=[]
    for b in bundles:
        groups={}
        for plan in b['plans']:
            rec=plan['price_record']; r=rec['r']; t=r%144
            scopes=[('midnight_today',0,144),('midnight_tomorrow',144,288)] if t==0 and plan['stage']!='root' else ([('intraday_next_hour',0,6)] if t in (36,72,108) else [])
            for label,a,z in scopes:
                ids=np.array(rec['targets'][a:z],int)
                if not len(ids): continue
                nodes=ids if rec['mapping']=='Q42' else np.maximum(ids-1,0); err=np.array(rec['prices'][a:z])-raw['price'][nodes]
                groups.setdefault(label,[]).extend(err.tolist())
        for scope,e in groups.items(): metrics.append(dict(branch=b['name'],method=b['method'],start=b['start'],scope=scope,count=len(e),MAE=float(np.mean(abs(np.array(e)))),RMSE=float(np.sqrt(np.mean(np.array(e)**2)))))
    with (STATE/'forecast_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0])); w.writeheader(); w.writerows(metrics)
    sources.verify(); mapping=export_boundaries(bundles)
    result=dict(status='PASS_REQUIRED_SMALL_VALIDATION',branches=reports,recovery=recovery,causality=causal,export_boundary_fixtures=mapping,forecast_metrics=metrics,coefficients=prices.coeffs,source_sha256=sources.hashes,base_sources={str(k):v for k,v in bases.refs.items()},risk_cache=bases.validation,annual_started=False,formal_result_workbooks_created=False)
    write(STATE/'small_validation.json',result); return result
if __name__=='__main__':
    stage=sys.argv[1] if len(sys.argv)>1 else ''; assert stage in ('prepare','pilot','full'), 'Only prepare/pilot/full-small modes exist'
    if (STATE/f'stage_{stage}.json').exists(): raise RuntimeError('Stage already recorded. Do not silently rerun or reset budget.')
    budget=Budget(stage,seconds=float(os.environ.get('Q4_TEST_SECONDS','110'))); sources=Sources(); status='FAILED'
    try:
        data=load_inputs(sources); bases=Bases(data,sources); prices=Prices(budget); identity=freeze(sources); hook_q2(budget)
        if stage!='prepare':
            previous=read(STATE/'prepared.json'); assert previous['status']=='PASS'
            for p,h in previous['source_sha256'].items():
                if sha(p)!=h: raise ValueError('Frozen source changed since preparation: '+p)
        if stage=='full':
            assert read(STATE/'pilot_report.json')['status']=='PASS_WIRING_PILOT'
        result=globals()[stage](data,sources,budget,bases,prices,identity); status='PASS'; write(STATE/f'{stage}_outcome.json',result)
        print(json.dumps(dict(stage=stage,status=status,budget=budget.record(),estimate=result.get('full_test_estimate_seconds')),ensure_ascii=False),flush=True)
    except BaseException:
        write(STATE/f'{stage}_failure.json',dict(error=traceback.format_exc(),budget=budget.record())); raise
    finally:
        write(STATE/f'stage_{stage}.json',dict(budget.record(),status=status)); write(STATE/f'{stage}_sources.json',sources.hashes)
