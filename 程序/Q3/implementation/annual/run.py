"""User-started annual job; importing or preflight never starts control."""
import os,time,sys
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[_key]='1'
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from runtime import HERE,read,sha,atomic,Budget,BaseCache,StopRun
import argparse,traceback,platform,copy
import numpy as np
import scipy
from data_access import read_inputs,Access
from control import AnnualController
from independent import raw_sources,audit_block,close
from reporting import finish

PACKAGE=HERE.parent.parent; ROOT=next(p for p in Path(__file__).resolve().parents if (p/'C题'/'附件'/'附件1.xlsx').is_file()); SUCCESS=ROOT/'核验依据/Q3_小验证'
CORE_NAMES=('core.py','controller.py','forecasting.py','data_access.py','audit.py')

def identity():
    protocol=read(SUCCESS/'protocol.json')
    if read(SUCCESS/'summary.json')['status']!='PASS_SMALL_E1_LP10': raise ValueError('Small validation missing')
    for name in CORE_NAMES:
        if sha(HERE.parent/name)!=protocol['code_sha256'][name] or sha(SUCCESS/'frozen_source'/name)!=protocol['code_sha256'][name]: raise ValueError('Verified core drift: '+name)
    code={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file() and p.suffix in ('.py','.ps1')}
    pins=read(HERE/'release_identity.json')
    if pins['code_sha256']!=code: raise ValueError('Annual implementation changed since regression freeze')
    if scipy.__version__!=protocol['solver']['scipy'] or np.__version__!=protocol['solver']['numpy'] or sys.version!=protocol['solver']['python']: raise ValueError('Solver/runtime differs from verified configuration')
    return dict(model='E1-LP10',source_sha256=protocol['source_sha256'],prediction_sha256=protocol['code_sha256']['forecasting.py'],core_sha256={k:protocol['code_sha256'][k] for k in CORE_NAMES},code_sha256=code,protocol_sha256={p.name:sha(p) for p in (PACKAGE/'02_Codex完整求解交接.md',PACKAGE/'03_结果账本与导出冻结约束.md',PACKAGE/'06_小验证通过后的实施交接.md')},solver=protocol['solver'])

def authorize(approved,regression):
    if not approved: raise ValueError('Annual mode requires explicit --approve-annual from user launch')
    if regression.get('status')!='PASS_WIRING' or regression.get('annual_started') is not False: raise ValueError('Matching wiring regression not passed')

def commit_day(out,day,controllers,raw,budget,ident,previous):
    files={}; audits={}; states={}
    for policy in ('C2','C0','C1'):
        c=controllers[policy]; tick=time.perf_counter()
        initial=6000. if day==0 else previous['states'][policy]['state']['energy']
        result=audit_block(c.block(),raw,initial,start=day*144,end=(day+1)*144,cache_dir=out/'cache')
        budget.addtime('independent_daily_audit',time.perf_counter()-tick)
        if day<2:
            old=read(SUCCESS/policy/'complete.json')['state']['ledger'][day*144:(day+1)*144]
            fields=('q0','a_final','c','s','g','w','E_before','E_after','normal','adjustment','emergency','total','R')
            for a,b in zip(c.ledger,old): close([a[k] for k in fields],[b[k] for k in fields])
        block=c.block(); block['audit']=result
        rel=f'days/{day:03d}/{policy}.json.gz'
        files[rel]=budget.write(out/rel,block); audits[policy]=result; states[policy]=c.state()
    prior_hash=previous.get('commit_hash') if previous else None
    cp=dict(identity=ident,next_day=day+1,states=states,files=files,previous_commit=prior_hash,resources=budget.record(),audits=audits)
    rel=f'commits/{day:03d}.json'; h=budget.write(out/rel,cp)
    budget.write(out/'checkpoint.json',dict(commit_file=rel,commit_hash=h))
    cp['commit_hash']=h
    for c in controllers.values(): c.clear_day()
    return cp

def load_checkpoint(out,ident):
    head=read(out/'checkpoint.json'); path=out/head['commit_file']
    if sha(path)!=head['commit_hash']: raise ValueError('Checkpoint hash mismatch')
    cp=read(path)
    if cp['identity']!=ident: raise ValueError('Resume identity mismatch')
    for rel,h in cp['files'].items():
        if sha(out/rel)!=h: raise ValueError('Committed day changed')
    cp['commit_hash']=head['commit_hash']; return cp

def previous_seconds(out):
    log=out/'process_sessions.jsonl'
    if not log.exists(): return 0.
    records=[__import__('json').loads(s) for s in log.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    if any(r.get('wall_seconds') is None for r in records): raise ValueError('Incomplete external timing; cumulative budget cannot be reset')
    return sum(r['wall_seconds'] for r in records)

def annual(args):
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    limits=dict(wall_seconds=args.wall_seconds,output_bytes=int(args.output_gib*2**30),calls=args.max_calls,fits=args.max_fits,processes=1,solver_threads=1,single_lp_seconds=5)
    if not 0<args.wall_seconds<=900 or not 0<args.output_gib<=2 or not 0<args.max_calls<=27000 or not 0<args.max_fits<=1460: raise ValueError('Limits exceed frozen annual protocol')
    ident=identity(); accepted=HERE/'wiring_acceptance.json'; regression=read(accepted) if accepted.exists() else {}
    authorize(args.approve_annual,regression)
    if regression['identity']!=ident: raise ValueError('Regression identity mismatch')
    if sha(Path(regression['evidence'])/'summary.json')!=regression['summary_sha256']: raise ValueError('Regression evidence changed')
    lock=out/'run.lock'
    # Advisory lock survives file presence but is released by the OS after a crash.
    import msvcrt
    handle=lock.open('a+b'); handle.seek(0); handle.write(b'0'); handle.flush(); handle.seek(0)
    try: msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
    except OSError: handle.close(); raise ValueError('Another process owns this output')
    b=None; controllers={}
    try:
        manifest=out/'manifest.json'
        if args.resume:
            old=read(manifest)
            if old['identity']!=ident or old['limits']!=limits: raise ValueError('Resume cannot change identities or cumulative limits')
            if (out/'status.json').exists() and read(out/'status.json')['status']=='INVALID': raise ValueError('Invalid numerical/information run cannot be resumed')
        elif manifest.exists() or (out/'resource_events.jsonl').exists():
            status=read(out/'status.json').get('status') if (out/'status.json').exists() else None
            if status=='INVALID': raise ValueError('Existing run is INVALID: retain its evidence and choose a NEW output directory after repair; --resume is not allowed')
            raise ValueError('Output already contains a run: use --resume only for a valid interrupted run, or select a new output directory')
        b=Budget(out,limits,args.launch_epoch,previous_seconds(out)); b.check()
        if not args.resume:
            b.write(manifest,dict(identity=ident,limits=limits,start_policy='Jan1 E6000 per branch; recompute two-day annual prefix',annual_started=True,formal_export_allowed=False,small_evidence=str(SUCCESS),wiring_evidence=regression))
            for p in HERE.iterdir():
                if p.suffix in ('.py','.ps1','.json'): b.write(out/'frozen_source'/(p.name+'.text.json'),dict(filename=p.name,text=p.read_text(encoding='utf-8')))
        tick=time.perf_counter(); data=read_inputs(ROOT,ident); raw=raw_sources(ROOT,ident['source_sha256']); b.addtime('input_and_independent_raw',time.perf_counter()-tick)
        cache=BaseCache(out/'cache',dict(source_sha256=ident['source_sha256'],prediction_sha256=ident['prediction_sha256']),b)
        cp=load_checkpoint(out,ident) if args.resume and (out/'checkpoint.json').exists() else None
        day=cp['next_day'] if cp else 0
        for policy in ('C2','C0','C1'):
            c=AnnualController(Access(data,policy,day*144),b,ident,cache)
            if cp: c.restore(cp['states'][policy])
            controllers[policy]=c
        speeds=read(out/'day_timings.json') if (out/'day_timings.json').exists() else {}
        # A Jan31 gate stop remains a stop on resume; no silent budget/gate override.
        if day==31 and (out/'january_gate.json').exists() and not read(out/'january_gate.json')['passed']: raise StopRun('Jan31 speed gate previously stopped this run')
        for day in range(day,365):
            tick=time.perf_counter()
            for policy,c in controllers.items():
                while c.position<(day+1)*144: c.step()
            cp=commit_day(out,day,controllers,raw,b,ident,cp)
            speeds[str(day)]=time.perf_counter()-tick; b.write(out/'day_timings.json',speeds)
            print(f'Day {day+1}/365 committed; LP={b.calls}, fits={b.fits}, elapsed={b.elapsed():.2f}s',flush=True)
            if day==30:
                recent=[speeds[str(d)] for d in range(21,31)]
                estimate=334*max(2*sum(recent)/10,max(recent))+30
                gate=dict(passed=b.elapsed()<=120 and b.elapsed()+estimate<=limits['wall_seconds'],elapsed=b.elapsed(),remaining_estimate=estimate,estimated_finish=b.elapsed()+estimate,conservative_only=True,plans_each=738,segments_each=4464,Feb1_SOC={p:c.energy for p,c in controllers.items()})
                b.write(out/'january_gate.json',gate)
                if not gate['passed']: raise StopRun('Jan31 speed gate: checkpoint retained; no automatic extension')
        tick=time.perf_counter(); answer=finish(out,raw,ident,b); b.addtime('final_audit_and_delivery',time.perf_counter()-tick)
        if identity()!=ident: raise ValueError('Code/protocol changed during annual run')
        from run_small import peak_memory
        b.write(out/'status.json',dict(status='PASS_ANNUAL_DRAFT_MAPPING',annual_completed=True,formal_export_allowed=False,python_peak_working_set_bytes=peak_memory(),resources=b.record(),results=answer))
        b.write(out/'resource_progress.json',b.record()); return 0
    except BaseException as exc:
        if b:
            status='STOPPED' if isinstance(exc,(StopRun,KeyboardInterrupt)) else 'INVALID'
            atomic(out/'status.json',dict(status=status,error=traceback.format_exc(),annual_completed=False,formal_export_allowed=False,resources=b.record()))
            # Emergency evidence is allowed after a limit, never new optimization.
            for policy,c in controllers.items(): atomic(out/'failure'/f'{policy}.json.gz',dict(state=c.state(),uncommitted=c.block()))
            atomic(out/'resource_progress.json',b.record())
        raise
    finally:
        handle.seek(0); msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1); handle.close()

def parser():
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=('preflight','annual'),default='preflight'); p.add_argument('--approve-annual',action='store_true'); p.add_argument('--resume',action='store_true'); p.add_argument('--output')
    p.add_argument('--wall-seconds',type=float,default=900); p.add_argument('--output-gib',type=float,default=2); p.add_argument('--max-calls',type=int,default=27000); p.add_argument('--max-fits',type=int,default=1460); p.add_argument('--launch-epoch',type=float)
    return p
def main():
    args=parser().parse_args()
    if args.mode=='preflight':
        ident=identity(); read_inputs(ROOT,ident); print('PREFLIGHT PASS; no fit, optimization or annual control'); return 0
    if not args.output: raise ValueError('--output is required')
    return annual(args)
if __name__=='__main__': sys.exit(main())
