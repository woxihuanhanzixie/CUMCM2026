"""Single finite wiring suite: 9 LP, 4 fits, 36 executed segments; no full day."""
import os,sys,time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from runtime import HERE,read,sha,atomic,Budget,BaseCache
import copy,traceback,argparse
import numpy as np
from core import Problem,solve_lp
from data_access import Access,read_inputs
from control import AnnualController
from independent import close,raw_sources,audit_block,audit_base_values
from run import identity,ROOT,SUCCESS,authorize,load_checkpoint

def reject(fn):
    try: fn()
    except (ValueError,AssertionError,RuntimeError): return
    raise AssertionError('Expected rejection did not occur')

def migrate_fixture(record,ident):
    old=read(SUCCESS/'protocol.json')
    if record['identity']['source_sha256']!=ident['source_sha256'] or record['identity']['code_sha256']!=old['code_sha256'] or record['identity']['protocol_sha256']!=sha(SUCCESS/'protocol.json'): raise ValueError('Historical checkpoint identity')
    if record['state']['position']!=174: raise ValueError('Only approved i174 fixture may migrate')
    # Explicit evidence migration for tests only, never annual resume.
    record=copy.deepcopy(record); record['identity']=ident; return record

def static_checks(out,ident,b):
    reject(lambda:authorize(False,dict(status='PASS_WIRING',annual_started=False)))
    reject(lambda:authorize(True,dict(status='FAIL',annual_started=False)))
    from control import AnnualController
    from controller import Controller
    import inspect
    old=inspect.getsource(Controller.step)
    new=inspect.getsource(AnnualController.step)
    expected=old.replace("if i>=288 or self.access.position!=i: raise ValueError('Small entry limited to Jan1-Jan2')","if i>=self.end_position or self.access.position!=i: raise ValueError('Outside configured annual domain')")
    if new.strip()!=expected.strip(): raise AssertionError('Step changed beyond domain guard')
    for slot in (0,36,72,108,138):
        count=min((364+2)*144,52560)-(364*144+slot)
        if count!=144-slot: raise AssertionError('Year-end range clipping')
    fake=out/'atomic_check'; fake.mkdir(); h=atomic(fake/'commits/000.json',dict(identity=ident,next_day=1,files={},states={}))
    atomic(fake/'checkpoint.json',dict(commit_file='commits/000.json',commit_hash=h))
    if load_checkpoint(fake,ident)['next_day']!=1: raise AssertionError('Commit readback')
    atomic(fake/'commits/001.json',dict(uncommitted=True))
    if load_checkpoint(fake,ident)['next_day']!=1: raise AssertionError('Uncommitted block advanced head')
    bad=copy.deepcopy(ident); bad['model']='wrong'; reject(lambda:load_checkpoint(fake,bad))
    atomic(fake/'commits/000.json',dict(changed=True)); reject(lambda:load_checkpoint(fake,ident))
    tiny=Budget(out/'limit_check',dict(wall_seconds=120,output_bytes=2**20,calls=0,fits=0))
    reject(lambda:tiny.reserve('lp','must_not_solve')); reject(lambda:tiny.reserve('fit','must_not_fit'))
    return dict(status='PASS',unchanged_step_except_domain=True,authorization_rejections=2,atomic_commit_checks=4,year_end_shapes=5,budget_rejections=2)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True); parser.add_argument('--launch-epoch',type=float); args=parser.parse_args()
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists() or (out/'resource_events.jsonl').exists(): raise ValueError('Do not rerun a wiring budget directory')
    b=Budget(out,dict(wall_seconds=120,output_bytes=256*2**20,calls=12,fits=4),args.launch_epoch); results={}
    try:
        ident=identity(); b.write(out/'identity.json',ident)
        results['static']=static_checks(out,ident,b)
        tick=time.perf_counter(); data=read_inputs(ROOT,ident); raw=raw_sources(ROOT,ident['source_sha256']); b.addtime('input',time.perf_counter()-tick)
        cacheident=dict(source_sha256=ident['source_sha256'],prediction_sha256=ident['prediction_sha256']); cache=BaseCache(out/'cache',cacheident,b)
        # Read-only fixture structure checks before spending any solver budget.
        fixtures={p:migrate_fixture(read(SUCCESS/p/'checkpoint_i174.json'),ident) for p in ('C0','C1','C2')}
        saved={p:read(SUCCESS/p/'complete.json')['state'] for p in fixtures}
        cases=[read(SUCCESS/'V1'/f'{name}.json') for name in ('i144_order','i36_adjust','i180_feedback')]
        for item in cases: Problem(**item['problem'])
        replay={}
        for policy,cp in fixtures.items():
            c=AnnualController(Access(data,policy,174),b,ident,cache); c.restore(cp)
            # Serialize and restore annual compact state before execution.
            state=c.state(); b.write(out/f'{policy}/compact_i174.json',state)
            fresh=AnnualController(Access(data,policy,174),b,ident,cache); fresh.restore(read(out/f'{policy}/compact_i174.json'))
            # Fixture prior-day evidence is test-only; annual midnight resume needs no old blocks.
            fresh.forecasts=c.forecasts; fresh.versions=c.versions; fresh.access.log=c.access.log
            while fresh.position<186: fresh.step()
            expected=saved[policy]['ledger'][174:186]
            for a,z in zip(fresh.ledger,expected):
                for k in ('q0','a_final','c','s','g','w','E_before','E_after','R','normal','adjustment','emergency','total'): close(a[k],z[k])
                for k in ('final_version_id','final_published_at','official_version','history_cutoff','hour_plan_time','delivery_start'):
                    if str(a[k])!=str(z[k]) and not (k=='official_version' and list(a[k])==list(z[k])): raise AssertionError('Replay metadata differs: '+k)
            block=fresh.block(); block['access_log']=[e for e in block['access_log'] if 144<=e['position']<186]; block['forecasts']=[f for f in block['forecasts'] if f['position']>=144]
            audit=audit_block(block,raw,cp['state']['energy'],174,186)
            # Audit corruption checks without new numerical control.
            corrupted=copy.deepcopy(block); corrupted['ledger'][0]['total']+=.1; reject(lambda:audit_block(corrupted,raw,cp['state']['energy'],174,186))
            corrupted=copy.deepcopy(block); corrupted['ledger'][0]['source_node_index']+=1; reject(lambda:audit_block(corrupted,raw,cp['state']['energy'],174,186))
            b.write(out/f'{policy}/replay.json.gz',block); replay[policy]=dict(status='PASS',segments=12,LP=2,max_numeric_error=max(abs(a[k]-z[k]) for a,z in zip(fresh.ledger,expected) for k in ('q0','a_final','E_after','total')),audit=audit)
        results['replay']=replay
        fixed=[]
        for item in cases:
            sol=solve_lp(Problem(**item['problem']),b,'wiring_'+item['name'])
            for key in ('A','E','upper','lower'): close(sol[key],item['candidate'][key])
            b.write(out/'fixed'/f'{item["name"]}.json',dict(problem=item['problem'],solution=sol)); fixed.append(item['name'])
        results['fixed_LP']=dict(status='PASS',cases=fixed)
        d=31; first=cache.get(Access(data,'C2',d*144),d)
        if b.fits!=4: raise AssertionError('Feb1 base fits must be four')
        for policy in ('C0','C1'):
            hit=BaseCache(out/'cache',cacheident,b).get(Access(data,policy,d*144),d)
            if hit!=first: raise AssertionError('Disk shared cache identity or values')
        for variable in ('load','pv'):
            for k in (0,1): close(first['predictions'][variable][k*144:(k+1)*144],read(SUCCESS/'V3'/f'base_d31_k{k}_{variable}.json')['prediction'])
        poison=copy.deepcopy(data)
        for var in ('load_nodes','pv_nodes'): poison[var][d*144-1:]=np.nan
        hit=cache.get(Access(poison,'C0',d*144),d)
        if hit!=first or b.fits!=4: raise AssertionError('Future poison or cache refit')
        badpath=out/'cache'/(first['key']+'.json.gz'); original=read(badpath); altered=copy.deepcopy(original); altered['history_hashes']['load']='bad'; atomic(badpath,altered)
        reject(lambda:BaseCache(out/'cache',cacheident,b).get(Access(data,'C0',d*144),d)); atomic(badpath,original)
        results['cache']=dict(status='PASS',fits=4,other_policy_extra_fits=0,future_poison=True,history_identity_rejection=True,fixed_predictions_matched=4)
        audit_base_values(first,raw)
        # New CSV mapping writer only; reuse previously frozen synthetic values.
        from reporting import draft_maps
        from datetime import datetime,timedelta
        synthetic=read(SUCCESS/'V6_synthetic_ledger.json')['ledger']; mapped=[]
        for r in synthetic:
            d=363+r['day']; s=r['slot']; published=d*144+(s//36)*36
            mapped.append(dict(delivery_start=r['delivery'],delivery_end=(datetime.fromisoformat(r['delivery'])+timedelta(minutes=10)).isoformat(),day=d,slot=s,q0=r['q0'],a_final=r['a_final'],price=r['p'],normal=r['p']*r['a_final'],adjustment=.5*r['p']*abs(r['a_final']-r['q0']),emergency=5*r['p']*r['g'],c=r['c'],s=r['s'],g=r['g'],E_before=r['E_before'],E_after=r['E_after'],q0_created_at=d*144,final_version_id=f'SYNTHETIC_d{d}_s{(s//36)*36}',final_published_at=published))
        draft_maps(mapped,out/'synthetic_mapping',b,363,True)
        results['new_mapping']=read(out/'synthetic_mapping/DRAFT/mapping_readback.json')
        # Saved Jan5 failure: reproduce the exact adjustment LP and validate next
        # locked-hour Problem after normalization, without another real day.
        failed=SUCCESS.parent/'annual_user_20260913_01'
        problem=read(failed/'current_problem.json')['problem']
        cp=read(failed/'failure/C2.json.gz')['state']
        sol=solve_lp(Problem(**problem),b,'repair_saved_i684_adjust')
        c=AnnualController(Access(data,'C2',684),b,ident,cache)
        c.q0=np.array(cp['state']['q0']); c.a=np.array(cp['state']['a'])
        c.publish(4,108,sol['A'][:36],'adjust')
        if min(c.a)<0 or np.max(abs(c.a-np.array(cp['state']['a'])))>1e-9: raise AssertionError('Numerical-zero repair changed contract materially')
        repaired=c.a.copy(); state=copy.deepcopy(cp); state['identity']=ident
        c=AnnualController(Access(data,'C2',690),b,ident,cache); c.restore(state); c.a=repaired
        next_problem=c.problem(4,114,'feedback'); next_solution=solve_lp(next_problem,b,'repair_saved_i690_feedback')
        results['negative_zero_repair']=dict(status='PASS',raw_min=float(min(sol['A'])),normalized_min=float(min(repaired)),next_hour_gap=abs(next_solution['upper']-next_solution['lower']),extra_LP=2,extra_real_segments=0)
        b.write(out/'negative_zero_repair.json',dict(original_problem=problem,adjustment_solution=sol,repaired_contract=repaired,next_problem=next_problem.record(),next_solution=next_solution))
        bad=np.zeros(36); bad[0]=-1e-5
        reject(lambda:c.publish(4,108,bad,'adjust'))
        if b.calls!=11 or b.fits!=4: raise AssertionError('Wrong finite regression count')
        b.check(); results.update(status='PASS_WIRING',identity=ident,annual_started=False,real_segments=36,independent_MILP_calls=0,resource=b.record())
        from run_small import peak_memory
        results['python_peak_working_set_bytes']=peak_memory()
        b.write(out/'summary.json',results); atomic(HERE/'wiring_acceptance.json',dict(status='PASS_WIRING',identity=ident,annual_started=False,evidence=str(out),summary_sha256=sha(out/'summary.json')))
        print(__import__('json').dumps(dict(status=results['status'],output=str(out),resources=b.record()),ensure_ascii=False)); return 0
    except BaseException:
        results.update(status='FAIL_WIRING',annual_started=False,error=traceback.format_exc(),resource=b.record()); atomic(out/'summary.json',results); print(results['error'],flush=True); return 1
if __name__=='__main__': sys.exit(main())
