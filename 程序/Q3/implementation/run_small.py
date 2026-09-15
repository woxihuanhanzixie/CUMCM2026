"""One bounded, frozen V1-V6 run. Failures are retained; never launches annual work."""
import os,time
START=time.perf_counter()
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
import sys,json,hashlib,traceback,copy,subprocess,ctypes
from pathlib import Path
from datetime import datetime
import numpy as np
import scipy
from core import Problem,Budget,solve_lp,reference_milp,execute,save,digest,LO,HI,P
from forecasting import base_predict,fused_curve,initial_base
from data_access import read_inputs,Access,shared_bytes
from controller import Controller
from audit import physical,raw_audit_sources,audit_controller
from export_check import export_check

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]; OLD=ROOT/'逐问修订/Q3/Astra_v5_Codex验证执行交接_20260912'
SOURCES={'C题/C题.pdf':'2c098f6ae9dd47ae965aebdec3b9facf3de6c173f999783c1012c08fc5fb9d2d','C题/附件/附件1.xlsx':'66b87134f5ecccd68184d3539bb1293ef039f9e0fdd955a589b9bfa7f227c377','C题/附件/附件2.xlsx':'2e95fd446bfafa0d8c59577b5c2e2ea8b3f1def20dde54a3062556f4da9b4c72','C题/附件/附件3.xlsx':'8a61b06c52bd0d639a1cc37c61a7d9f5b75edcbca718f64c1bd3498ec9f9d843','C题/附件/附件5/result3.xlsx':'c59da470cabd0be23f602c95c8aa9d11ec224a0cdac216b3e1f218e65d006bdc'}
REUSED={'forecasting.py':'a2bf600a1ee0ce9bed13f27d1a1faffeb652e8d90f1ae0e09cd677bf5cce97d8','data_access.py':'0fa1a45eded75fc582e8ad10e12a22d5cf2c32d6b283c6f83b960e191b6da454'}
def sha(p): return hashlib.sha256(shared_bytes(Path(p))).hexdigest()
def expect_rejection(f):
    try: f()
    except (ValueError,AssertionError): return
    raise AssertionError('Illegal input/access was accepted')
def maxdiff(a,b):
    if isinstance(a,dict):
        if set(a)!=set(b): raise AssertionError('Different record keys')
        return max([maxdiff(a[k],b[k]) for k in a]+[0.])
    if isinstance(a,(list,tuple,np.ndarray)):
        if len(a)!=len(b): raise AssertionError('Different record length')
        return max([maxdiff(x,y) for x,y in zip(a,b)]+[0.])
    if isinstance(a,(float,int,np.number)) and not isinstance(a,bool): return abs(float(a)-float(b))
    if a!=b: raise AssertionError(('Metadata mismatch',a,b))
    return 0.

def fixtures():
    def p(ell,pv=None,energy=LO,stage='original',q=None,a=None,prices=None):
        n=len(ell); return dict(ell=ell,pv=[0]*n if pv is None else pv,prices=[1]*n if prices is None else prices,energy=energy,today=n,stage=stage,q0=q,current=a)
    cases=[('anti_equal',p([50,100,150,200,250,300])),('early_deficit',p([450,450,0,0,0,0],energy=2200,stage='feedback',a=[0]*6)),('sequence_deficit_surplus',p([300,300,300,0,0,0],pv=[0,0,0,300,300,300],stage='feedback',a=[0]*6)),('adjust_large_original',p([100,200,50],pv=[900,0,500],energy=6000,stage='adjust',q=[2000,50,1800],a=[2200,60,1900],prices=[.4,1.4,.8])),('locked_upper_SOC',p([0,1200,0],pv=[500,0,2000],energy=HI,stage='feedback',a=[200,0,0])),('equal_balance',p([100,200],pv=[100,200],energy=HI,stage='feedback',a=[0,0])),('free_surplus_lower_SOC',p([0,0,100],pv=[1500,1500,0])),('free_price_shift',p([1000,1000,1500,500],pv=[0,600,0,0],energy=6000,prices=[.3,.3,1.4,.8]))]
    for name in ('i36_adjust','i144_order','i180_feedback','i252_feedback'):
        cases.append((name,json.loads((OLD/f'outputs/T2/20260913_003857_635414/solver/{name}_input.json').read_text(encoding='utf-8'))))
    return cases

def v1_v2(cases,b):
    results=[]; property_count=0; maxres=0
    for name,raw in cases:
        p=Problem(**raw); candidate=solve_lp(p,b,'V1_'+name); reference=reference_milp(p,b,'V1_ref_'+name)
        delta=abs(candidate['upper']-reference['upper'])
        record=dict(name=name,problem=raw,candidate=candidate,reference=reference,objective_difference=delta)
        save(b.out/f'V1/{name}.json',record); results.append(record)
        if delta>1e-6: raise AssertionError(('V1 objective difference',name,delta))
        # Conditional property uses the identical physical inputs and native states.
        E=p.energy
        for t in range(len(p.ell)):
            act=execute(E,candidate['A'][t],p.ell[t],p.pv[t],candidate['E'][t]); maxres=max(maxres,physical(E,p.ell[t],p.pv[t],candidate['A'][t],act))
            if act['E']<candidate['E'][t]-1e-6 or act['g']>candidate['g'][t]+1e-6: raise AssertionError(('Conditional property',name,t))
            E=act['E']; property_count+=1
    anti=results[0]['candidate']
    if maxdiff(anti['A'],[50,100,150,200,250,300])>1e-6: raise AssertionError('Hourly equalization detected')
    for key in ('c','s','g','w'):
        if np.max(abs(anti[key]))>1e-6: raise AssertionError('Anti-equal fixture physical expectation')
    if abs(results[1]['candidate']['g'].sum())>1e-6 or maxdiff(results[1]['candidate']['s'][:2],[450,450])>1e-6: raise AssertionError('Early-deficit counterexample')
    if abs(results[2]['candidate']['g'].sum()-900)>1e-6: raise AssertionError('Chronological deficit counterexample')
    for k in (0,8):
        repeat=solve_lp(Problem(**cases[k][1]),b,'V1_repeat_'+cases[k][0])
        if maxdiff(repeat['A'],results[k]['candidate']['A'])>1e-6 or maxdiff(repeat['E'],results[k]['candidate']['E'])>1e-6: raise AssertionError('Deterministic repeat')
    illegal=copy.deepcopy(cases[0][1]); illegal['prices'][0]=0; expect_rejection(lambda:Problem(**illegal))
    illegal['prices'][0]=-1; expect_rejection(lambda:Problem(**illegal))
    rng=np.random.default_rng(20260913); stress=[]
    for E in [LO,HI,6000.]:
        for R in [LO,HI,8000.]:
            for ell,pv,A in [(5000,0,0),(0,5000,0),(0,0,0),(100,0,100),(1200,0,0)]+[tuple(rng.uniform(0,4000,3)) for _ in range(10)]:
                act=execute(E,A,ell,pv,R); maxres=max(maxres,physical(E,ell,pv,A,act)); stress.append(dict(E=E,R=R,ell=ell,pv=pv,A=A,action=act))
    save(b.out/'V2_stress.json',stress)
    return dict(V1=dict(status='PASS',pairs=12,max_objective_difference=max(r['objective_difference'] for r in results),anti_equal_orders=anti['A'],independence='independent physical formulation; same SciPy/HiGHS solver'),V2=dict(status='PASS',conditional_segments=property_count,stress_cases=len(stress),max_physical_residual=maxres,early_deficit_emergency=0,sequence_emergency=900))

def independent_ridge(history,d,k,load,b):
    # Reconstruct feature rows with explicit scalar/day loops, solve penalized normal equations.
    b.reserve('fit',f'independent_ridge_{d}_{k}_{load}')
    def feature(h,t):
        col=[history[h-1,t],history[h+k-7,t],history[h+k-14,t],sum(history[z,t] for z in range(h-3,h))/3,sum(history[z,t] for z in range(h-7,h))/7,sum(history[z,t] for z in range(h-3,h))/3-sum(history[z,t] for z in range(h-6,h-3))/3,float(np.mean(history[h-1]))]
        for j in (1,2,3): col.extend([np.sin(2*np.pi*j*t/144),np.cos(2*np.pi*j*t/144)])
        if load:
            from datetime import timedelta
            weekday=(datetime(2025,1,1)+timedelta(days=h+k)).weekday(); col.extend([float(weekday==x) for x in range(1,7)])
        return col
    hs=list(range(max(14,d-28-k),d-k)); X=np.array([feature(h,t) for h in hs for t in range(144)]); Y=np.array([history[h+k,t] for h in hs for t in range(144)]); w=np.array([2**(-(d-1-h-k)/14) for h in hs for t in range(144)])
    mu=np.average(X,axis=0,weights=w); sd=np.sqrt(np.average((X-mu)**2,axis=0,weights=w)); sd=np.where(sd<1e-12 if load else sd<=1e-12,1.,sd); Z=np.column_stack([np.ones(len(X)),(X-mu)/sd]); H=Z.T@(w[:,None]*Z)+np.diag([0.]+[1.]*(Z.shape[1]-1)); beta=np.linalg.solve(H,Z.T@(w*Y)); target=np.array([feature(d,t) for t in range(144)])
    return np.maximum(np.column_stack([np.ones(144),(target-mu)/sd])@beta,0)

def v3(data,b):
    tick=time.perf_counter(); records=[]; pred={}; hist={}
    for d in (21,31):
        access=Access(data,'C2',d*144)
        for k in (0,1):
            for variable in ('load','pv'):
                history=access.history_before(d,variable); y,r=base_predict(history,d,k,variable=='load',b); records.append(r); pred[d,k,variable]=y; hist[d,variable]=history
                save(b.out/f'V3/base_d{d}_k{k}_{variable}.json',dict(prediction=y,record=r,resource=b.record()))
    independent_diffs=[]
    for d,k,var in [(21,0,'load'),(21,0,'pv'),(31,1,'load'),(31,1,'pv')]:
        z=independent_ridge(hist[d,var],d,k,var=='load',b); delta=float(np.max(abs(z-pred[d,k,var]))); independent_diffs.append(delta)
        if delta>1e-6: raise AssertionError(('Independent ridge',d,k,var,delta))
    node_error=0.; checks=[]
    for day in (0,1,31,364):
        base=initial_base(data['official'][0,0]) if day==0 else np.linspace(0,1000,144 if day==364 else 288)
        for slot in (0,36,72,108):
            bulletin=data['official'][day,slot]; anchor=17.; curve=fused_curve(base,slot,bulletin,anchor,jan1=day==0)
            if curve[slot]!=anchor or len(curve)!=len(base): raise AssertionError('Fusion anchor/domain')
            for k in range(1,25):
                t=slot+6*k
                if t<len(curve):
                    e=abs(curve[t]-bulletin[k-1]); node_error=max(node_error,e)
                    if e>1e-7+1e-9*abs(bulletin[k-1]): raise AssertionError('Official node miss')
            if not np.array_equal(curve[:slot],base[:slot]) or not np.array_equal(curve[slot+145:],base[slot+145:]): raise AssertionError('Official coverage fallback')
            # Independent scalar interpolation, including finite end baseline left limit.
            for t in range(slot,min(len(base),slot+145)):
                lead=(t-slot)/6; left=int(lead); frac=lead-left; nodes=np.r_[anchor,bulletin]
                if frac==0: expected=nodes[left]
                elif day==0: expected=(1-frac)*nodes[left]+frac*nodes[left+1]
                else:
                    il=slot+6*left; ir=il+6; expected=max(0,base[t]+(1-frac)*(nodes[left]-base[il])+frac*(nodes[left+1]-base[min(ir,len(base)-1)]))
                if abs(expected-curve[t])>1e-7+1e-9*abs(expected): raise AssertionError('Scalar fusion mismatch')
            checks.append(dict(day=day,slot=slot,n=len(curve)))
    if not np.all(initial_base(data['official'][0,0])[:6]==data['official'][0,0][0]): raise AssertionError('Jan1 first hour')
    a=Access(data,'C0',180); expect_rejection(lambda:a.official_released_by(1,36)); expect_rejection(lambda:a.official_released_by(2,0)); expect_rejection(lambda:a.observations_completed_by(181,'load')); expect_rejection(lambda:a.reveal_current_after_commit(180))
    # Poison all not-yet-completed actual nodes; legal training histories remain identical.
    poison=copy.deepcopy(data); cutoff=31*144
    for key in ('load_nodes','pv_nodes'): poison[key][cutoff-1:]=np.nan
    a=Access(poison,'C2',cutoff)
    for var in ('load','pv'):
        if not np.array_equal(a.history_before(31,var),hist[31,var]): raise AssertionError('Future actual changed historical fit inputs')
    expect_rejection(lambda:base_predict(np.ones((364,144)),364,1,True,b))
    save(b.out/'V3_forecast_records.json',dict(fits=records,independent_prediction_errors=independent_diffs,fusion_checks=checks))
    b.addtime('V3_prediction_and_checks',time.perf_counter()-tick)
    return dict(status='PASS',base_fits=sum(r['method']=='ridge' for r in records),independent_fits=4,max_stationarity=max(r.get('stationarity',0) for r in records),max_independent_prediction_error=max(independent_diffs),max_node_error=node_error,fusion_cases=len(checks),future_history_poison_passed=True)

def v4_v5(data,b,identity,controllers):
    tick=time.perf_counter(); checkpoints={}; answers={}
    for policy in ('C0','C1','C2'):
        c=Controller(Access(data,policy),b,identity); controllers[policy]=c
        while c.position<288:
            if c.position==174:
                cp=c.snapshot(); checkpoints[policy]=cp; save(b.out/f'{policy}/checkpoint_i174.json',cp)
            c.step()
        io=time.perf_counter(); save(b.out/f'{policy}/complete.json',c.snapshot()); b.addtime('I_O',time.perf_counter()-io)
        print(f'{policy}: 288 segments, 42 LPs completed',flush=True)
    audit_start=time.perf_counter(); raw=raw_audit_sources(ROOT)
    for p,c in controllers.items(): answers[p]=audit_controller(c,raw)
    # Meaningful audit corruption checks, no optimization.
    injected=copy.deepcopy(controllers['C2'].ledger[180]); controllers['C2'].ledger[180]['source_node_index']+=1
    expect_rejection(lambda:audit_controller(controllers['C2'],raw)); controllers['C2'].ledger[180]=injected
    injected=copy.deepcopy(controllers['C2'].ledger[180]); controllers['C2'].ledger[180]['total']+=.1
    expect_rejection(lambda:audit_controller(controllers['C2'],raw)); controllers['C2'].ledger[180]=injected
    b.addtime('independent_audit_V4',time.perf_counter()-audit_start); b.addtime('V4_end_to_end_excluding_initial_input',time.perf_counter()-tick)
    save(b.out/'V4_independent_audit.json',answers)
    # Checkpoint loaded from disk: six hourly solves total, including release at i180.
    replay_errors=[]
    for policy in ('C0','C1','C2'):
        cp=json.loads((b.out/f'{policy}/checkpoint_i174.json').read_text(encoding='utf-8')); c=Controller(Access(data,policy,174),b,identity); c.restore(cp)
        while c.position<186: c.step()
        delta=maxdiff(c.ledger[174:186],controllers[policy].ledger[174:186]); replay_errors.append(delta)
        if delta>1e-6: raise AssertionError('Checkpoint replay differs')
        bad=copy.deepcopy(cp); bad['identity']['code_sha256']['core.py']='wrong'; expect_rejection(lambda:c.restore(bad))
    # Future data/bulletins are poisoned at i174; current six executed values stay legal.
    poison=copy.deepcopy(data)
    for key in ('load_nodes','pv_nodes'): poison[key][179:]=np.nan
    for (d,s) in poison['official']:
        if d*144+s>174: poison['official'][d,s]=np.full(24,np.nan)
    for policy in ('C0','C1'):
        c=Controller(Access(poison,policy,174),b,identity); c.restore(checkpoints[policy])
        while c.position<180: c.step()
        if maxdiff(c.ledger[174:180],controllers[policy].ledger[174:180])>1e-6: raise AssertionError('Future poison changed past decisions/actions')
    # Legal synthetic C1 release mechanism: same initial state, locked orders, two nonnegative official forecasts.
    base=np.zeros(288); old=np.zeros(24); new=np.zeros(24); new[0]=6000
    curves=[fused_curve(base,36,x,0.) for x in (old,new)]; paths=[]
    for k,curve in enumerate(curves):
        p=Problem(np.full(12,300.),curve[36:48]/6,np.r_[np.ones(6),np.full(6,1.4)],2200.,12,'feedback',np.zeros(12),np.zeros(12))
        paths.append(solve_lp(p,b,f'V5_C1_official_{k}'))
    difference=float(np.max(abs(paths[0]['E']-paths[1]['E'])))
    if difference<=1e-6 or maxdiff(paths[0]['A'],paths[1]['A'])>1e-6: raise AssertionError('C1 new bulletin not changing reference with locked orders')
    save(b.out/'V5_recovery_information.json',dict(replay_errors=replay_errors,checkpoint_identity_rejections=3,future_poison_replays=2,synthetic_officials=[old,new],synthetic_paths=paths,reference_difference=difference))
    return answers,dict(status='PASS',replay_hours=6,max_replay_error=max(replay_errors),future_poison_hours=2,synthetic_C1_reference_difference=difference)

def peak_memory():
    class PMC(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(x,ctypes.c_size_t) for x in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    pm=PMC(); pm.cb=ctypes.sizeof(pm)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True); psapi=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(PMC),ctypes.c_ulong]
    psapi.GetProcessMemoryInfo.restype=ctypes.c_int
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(pm),pm.cb): raise ctypes.WinError(ctypes.get_last_error())
    return pm.PeakWorkingSetSize

def main():
    out=HERE.parent/'outputs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'); out.mkdir(parents=True); b=Budget(out,START); summary={}; controllers={}
    (HERE.parent/'latest_run.txt').write_text(str(out),encoding='utf-8')
    try:
        for name,expected in REUSED.items():
            if sha(HERE/name)!=expected or sha(OLD/'implementation'/name)!=expected: raise ValueError('Reused source identity drift '+name)
        cases=fixtures(); save(out/'frozen_cases.json',cases)
        code={p.name:sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.mjs','.ps1')}
        protocol=dict(model='E1-LP10',seed=20260913,max_optimizations=200,max_fits=12,max_wall_seconds=1200,user_authorization="2026-09-13 validation no more than 20 minutes",max_single_solve_seconds=5,source_sha256=SOURCES,reused_source_sha256=REUSED,code_sha256=code,cases_sha256=sha(out/'frozen_cases.json'),physical_tolerance=1e-6,V1_fee_tolerance=1e-6,LP_gap_tolerance=1e-5,segment_bill_tolerance=1e-6,daily_bill_tolerance=1e-4,total_bill_tolerance=1e-3,policies=['C0','C1','C2'],real_days=['2025-01-01','2025-01-02'],V1_expected_calls=26,V4_expected_calls=126,V5_expected_calls=10,expected_total_calls=162,expected_fits=10,Jan22_k1='only six training days; both variables legally fall back, zero fits',failure_policy='save and stop; no budget reset, no retry',solver=dict(method='highs-ds',threads=1,presolve=True,scipy=scipy.__version__,numpy=np.__version__,python=sys.version),V6='synthetic workbook only, no formal export',annual_started=False)
        save(out/'protocol.json',protocol); identity=dict(source_sha256=SOURCES,code_sha256=code,protocol_sha256=sha(out/'protocol.json'))
        tick=time.perf_counter(); data=read_inputs(ROOT,protocol); b.addtime('input',time.perf_counter()-tick)
        summary.update(v1_v2(cases,b)); save(out/'stage_progress.json',summary); print('V1/V2 passed',flush=True)
        summary['V3']=v3(data,b); save(out/'stage_progress.json',summary); print('V3 passed',flush=True)
        summary['V4'],summary['V5']=v4_v5(data,b,identity,controllers)
        summary['V6']=export_check(ROOT,HERE,out,b); print('V6 passed',flush=True)
        if b.calls!=162 or b.fits!=10: raise AssertionError(('Unexpected resource counts',b.calls,b.fits))
        for rel,expected in SOURCES.items():
            if sha(ROOT/rel)!=expected: raise AssertionError('Original input changed')
        for name,expected in code.items():
            if sha(HERE/name)!=expected: raise AssertionError('Code changed during tests')
        b.check(); summary.update(status='PASS_SMALL_E1_LP10',annual_started=False,formal_export_allowed=False,year_end_contract_rule='UNRESOLVED',failure_count=0,resource=b.record(),python_peak_working_set_bytes=peak_memory())
    except Exception:
        summary.update(status='FAIL_OR_INVALID',failure_count=1,error=traceback.format_exc(),resource=b.record(),annual_started=False,formal_export_allowed=False)
        for name,c in controllers.items(): save(out/f'{name}/failure_checkpoint.json',c.snapshot())
        save(out/'failure.json',summary)
    summary['output_bytes_before_summary']=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    save(out/'summary.json',summary); save(out/'resource_ledger.json',b.record())
    save(out/'artifact_sha256.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact_sha256.json'})
    print(json.dumps(dict(output=str(out),status=summary['status'],calls=b.calls,fits=b.fits,wall_seconds=time.perf_counter()-b.start,error=summary.get('error')),ensure_ascii=False),flush=True)
    return 0 if summary['status']=='PASS_SMALL_E1_LP10' else 1
if __name__=='__main__': sys.exit(main())
