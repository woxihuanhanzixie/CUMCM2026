import argparse,sys,time,traceback,os,math,csv,json
from pathlib import Path
import numpy as np
from annual import *
from audit_kernel import audit,raw_sources,price_check

def check_day(b,raw,energy):
    rows=b['ledger']; start=rows[0]['position']; n=len(rows)
    report=audit(b,raw,n,energy)
    assert [p['position'] for p in b['plans'] if p['stage']=='root']==(list(range(start,start+n)) if b['name']=='R2' else [])
    assert [p['position'] for p in b['plans'] if p['stage']=='midnight']==[i for i in range(start,start+n) if i%144==0]
    # Compare R2 executed root to saved restored chosen solution; feasibility alone is insufficient.
    root={p['position']:p for p in b['plans'] if p['stage']=='root'}
    for r in rows:
        if r['position'] not in root: continue
        p=root[r['position']]
        if p['solution']['usable']:
            for k,v in [('c','c'),('s','s'),('g','z'),('w','w'),('E_after','E')]: assert abs(r[k]-p['restored'][v][0])<1e-6
    # Independent OLS on the completed raw history, distinct from scalar covariance implementation.
    seen=set()
    for p in b['plans']:
        rec=p['price_record']; c=rec['coefficient']
        if c is None or c['day'] in seen: continue
        seen.add(c['day']); ids=np.arange(c['first'],c['last']+1)
        net=(raw['load']-raw['pv'])/1000; x=net[ids]-net[ids-1008]; y=raw['price'][ids]-raw['price'][ids-1008]
        if np.sum((x-x.mean())**2)>1e-12:
            coef=np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),y,rcond=None)[0]
            assert max(abs(coef-np.array([c['a'],c['b']])))<1e-9
        e=y-c['a']-c['b']*x; den=math.fsum(float(t*t) for t in e[:-1]); rho=0 if den<=1e-12 else max(0,min(1,math.fsum(float(a*z) for a,z in zip(e[:-1],e[1:]))/den))
        assert abs(rho-c['rho'])<1e-9
    report['secondary_not_accepted']=sum(p['stage']=='root' and not p['solution'].get('secondary',{}).get('accepted',False) for p in b['plans'])
    return report

def setup(out,limits):
    budget=Budget(out,limits); ctl.hook_q2(budget); sources=Sources(); data=old.load_inputs(sources)
    bases=Bases(data,sources); prices=Prices(budget); jan=january(data,sources); identity=source_identity(sources)
    return budget,sources,data,bases,prices,jan,identity

def wiring(out):
    if out.exists(): raise ValueError('Use a new wiring output directory')
    prior=sorted(out.parent.glob('wiring_*/resources.json'),key=lambda p:p.stat().st_mtime)
    if prior:
        out.mkdir(parents=True); previous=read(prior[-1])
        if (prior[-1].parent/'failure.json').exists():
            previous['released_unused_after_clean_exception']=previous['lp']-previous['actual_lp']; previous['lp']=previous['actual_lp']
        save(out/'resources.json',previous)
        cache=prior[-1].parent/'price_cache.json'
        if cache.exists(): save(out/'price_cache.json',read(cache))
    started=time.perf_counter(); b,s,data,bases,prices,jan,identity=setup(out,dict(seconds=120,lp=100,fit=8,bytes=1024**3))
    raw=raw_sources(); checks={}; stats={}; forecast_count=0
    try:
        # All 334 published cache domains and source hashes, zero fits/optimizations.
        for day in range(31,365):
            view=View(data,'Q42','C2',day*144); vals=bases.get(view,day)
            assert len(vals[0])==(144 if day==364 else 288); forecast_count+=1
        checks['all_334_forecast_domains']=forecast_count
        for day in (31,32,33,364): bases.risk(View(data,'Q42','C2',day*144),day)
        checks['risk_Feb1_Feb2_Feb3_Dec31']=bases.validation
        checks['formal_initial_energy']=jan['final_energy']
        # Old same-input fixtures, 24 root decisions for a useful R2 timing sample.
        for branch in BRANCHES:
            existing=next((p.parent/(branch+'.q4z') for p in reversed(prior) if (p.parent/(branch+'.q4z')).exists() and p.parent.name!='wiring_01'),None)
            if existing:
                rr=unpack(existing); check_day(rr,raw,6000.); saved=read(PREREQ/f'work/full/{branch}_4464/result.json.gz')
                err=max(abs(float(r[k])-float(saved['ledger'][i][k])) for i,r in enumerate(rr['ledger']) for k in ('q0','c','s','g','w','E_after','total')); assert err<1e-6
                stats[branch]=dict(segments=len(rr['ledger']),seconds=2*sum(p['solution'].get('wall_s',0.) for p in rr['plans']),max_fixture_difference=err,bytes=existing.stat().st_size,reused_from=str(existing),timing_note='2x saved solver wall plus annual overhead reserve; direct final-stage pilot below')
                continue
            c=new_controller(branch,data,b,identity,bases,prices,energy=6000.)
            t0=time.perf_counter(); count=6
            for _ in range(count): c.step()
            elapsed=time.perf_counter()-t0; saved=read(PREREQ/f'work/full/{branch}_4464/result.json.gz')
            err=max(abs(float(c.ledger[i][k])-float(saved['ledger'][i][k])) for i in range(count) for k in ('q0','c','s','g','w','E_after','total'))
            assert err<1e-6; bb=bundle(c); path=out/(branch+'.q4z'); pack(path,bb)
            rr=unpack(path); check_day(rr,raw,6000.)
            # Fault injection in the saved ledger must be caught independently.
            bad=copy.deepcopy(rr); bad['ledger'][0]['total']+=1
            try: check_day(bad,raw,6000.)
            except AssertionError: pass
            else: raise AssertionError('bill corruption escaped audit')
            stats[branch]=dict(segments=count,seconds=elapsed,max_fixture_difference=err,bytes=path.stat().st_size)
        # Start formally from the independently replayed January state.
        c=new_controller('R2_P1',data,b,identity,bases,prices,energy=jan['final_energy']); c.step(); check_day(bundle(c),raw,10800.)
        # Resume from pre-midnight frozen evidence, preserving its real state and contract.
        oldsnap=read(PREREQ/'work/full/R2_P1_4464/before_midnight.json.gz')
        if 'payload' in oldsnap:
            assert digest(oldsnap['payload'])==oldsnap['sha256']; oldsnap=oldsnap['payload']
        if 'checkpoint' in oldsnap: oldsnap=oldsnap['checkpoint']
        pos=oldsnap['state']['position']; oldsnap['identity']=identity
        c=new_controller('R2_P1',data,b,identity,bases,prices,position=pos); c.load_checkpoint(oldsnap)
        c.ledger=[]; c.plans=[]; c.corrections=[]; c.access.log=[]
        c.step(); snap=c.checkpoint(); path=out/'midnight_resume.q4z'; pack(path,snap)
        restored=unpack(path); resumed=new_controller('R2_P1',data,b,identity,bases,prices,position=c.position); resumed.load_checkpoint(restored)
        before=c.energy; c.step(); resumed.step()
        err=max(abs(c.ledger[-1][k]-resumed.ledger[-1][k]) for k in ('q0','c','s','g','w','E_after','total')); assert err<1e-6
        checks['cross_midnight_saved_resume_max_error']=err
        # Dec31 midnight, next hour, and final interval. Explicit single-day domain.
        c=new_controller('R2_P1',data,b,identity,bases,prices,position=364*144,energy=6000.)
        c.step(); assert len(c.plans[0]['nbar'])==144 and len(c.plans[1]['nbar'])==144
        for slot in (6,143):
            c.position=364*144+slot; c.access.position=c.position; c.access.committed=None; c.step()
            assert len(c.plans[-1]['nbar'])==144-slot
        assert c.position==52560; checks['year_end_horizons']=[144,138,1]
        # Price vector replay under optimized daily coefficient caching.
        for plan in c.plans: price_check(plan['price_record'],raw)
        s.verify(); b.check()
        # Conservatively include daily serialize/audit/I/O as measured separately.
        sample=stats['R2_P1']; estimated=sample['seconds']/sample['segments']*48096+300
        projected_bytes=sample['bytes']/sample['segments']*48096+20*1024**2
        result=dict(status='PASS_Q42_ANNUAL_WIRING',checks=checks,stats=stats,estimated_seconds=estimated,estimated_output_bytes=projected_bytes,within_planned_budget=estimated<2400 and projected_bytes<1024**3,resources=b.record(),process_seconds=time.perf_counter()-started,identity=source_identity(s),source_hashes=s.hashes,annual_started=False,prior_attempts=[str(p) for p in prior])
        save(out/'acceptance.json',result); print(json.dumps({k:result[k] for k in ('status','checks','estimated_seconds','within_planned_budget','resources')},ensure_ascii=False,default=old.serial),flush=True)
    except BaseException:
        save(out/'failure.json',dict(traceback=traceback.format_exc(),resources=b.record())); raise

def size_check(out,b):
    n=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    if n>b.limits['bytes']: raise RuntimeError('BUDGET_STOP: output bytes exhausted')
    return n

def run(out,wiring_path,resume=False,stop_after=None):
    gate=read(wiring_path/'acceptance.json'); assert gate['status']=='PASS_Q42_ANNUAL_WIRING'
    ready=read(HERE/'outputs/ready.json'); assert ready['status']=='READY_FOR_USER_ANNUAL_RUN'
    assert ready['within_planned_budget'], 'Measured estimate exceeds Q42 allocation; user must coordinate budget'
    if out.exists() and not resume: raise ValueError('Output exists: use --resume for the same run')
    out.mkdir(parents=True,exist_ok=True); lock=out/'run.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.write(fd,str(os.getpid()).encode()); os.close(fd)
    b=None; c=None
    try:
        b,s,data,bases,prices,jan,identity=setup(out,LIMITS); raw=raw_sources()
        assert identity['code']==ready['identity']['code'] and identity['model_sha256']==gate['identity']['model_sha256'], 'Wiring/final engineering/code identity mismatch'
        for p,h in gate['source_hashes'].items(): assert sha(p)==h, 'Validated source changed: '+p
        manifest=out/'run_manifest.json'
        if manifest.exists(): assert read(manifest)['identity']==identity, 'Run identity changed'
        else: save(manifest,dict(identity=identity,run_status='RUNNING',initial_energy_source=jan))
        checkpoint=read(out/'checkpoint.json') if (out/'checkpoint.json').exists() else dict(next_day=31,energy={k:jan['final_energy'] for k in BRANCHES},chain='0'*64,commits=[])
        assert len(checkpoint['commits'])==checkpoint['next_day']-31
        # Verify every committed day, including source state links; pending uncommitted output may be replaced.
        chain='0'*64; energies={k:10800. for k in BRANCHES}
        for expected_day,ref in enumerate(checkpoint['commits'],31):
            path=out/ref['path']; assert sha(path)==ref['sha256']; cm=read(path); assert cm['previous']==chain
            assert cm['day']==expected_day
            for branch in BRANCHES:
                assert cm['branches'][branch]['initial_energy']==energies[branch]
                assert sha(out/cm['branches'][branch]['path'])==cm['branches'][branch]['sha256']
                energies[branch]=cm['branches'][branch]['final_energy']
            chain=ref['sha256']
        assert chain==checkpoint['chain'] and energies==checkpoint['energy']
        completed=0
        for day in range(checkpoint['next_day'],365):
            tick=time.perf_counter(); day_result={}
            for branch in BRANCHES:
                b.check(); c=new_controller(branch,data,b,identity,bases,prices,day*144,checkpoint['energy'][branch]); initial=c.energy
                for _ in range(144): c.step()
                bb=bundle(c); report=check_day(bb,raw,initial)
                rel=f'{branch}/days/{day:03d}.q4z'; h=pack(out/rel,bb)
                reread=unpack(out/rel); assert digest(reread)==digest(bb)
                day_result[branch]=dict(path=rel,sha256=h,initial_energy=initial,final_energy=c.energy,audit=report)
                clear_day(c)
            cm=dict(day=day,previous=checkpoint['chain'],branches=day_result,budget=b.record(),risk=[v for v in bases.validation if v['day']==day],forecast=bases.refs[('Q42',day)])
            rel=f'commits/{day:03d}.json'; h=save(out/rel,cm)
            checkpoint=dict(next_day=day+1,energy={k:v['final_energy'] for k,v in day_result.items()},chain=h,commits=checkpoint['commits']+[dict(path=rel,sha256=h)])
            save(out/'checkpoint.json',checkpoint); s.verify(); output_bytes=size_check(out,b)
            save(out/'status.json',dict(status='RUNNING',last_completed_day=day,next_day=day+1,resources=b.record(),output_bytes=output_bytes))
            print(f'{(datetime(2025,1,1)+timedelta(days=day)):%Y-%m-%d} complete | {time.perf_counter()-tick:.2f}s | cumulative {b.state["wall"]:.1f}s | LP {b.state["actual_lp"]}',flush=True)
            # Discard old prediction arrays; maintain metadata/source identities for audit.
            bases.hot.clear(); prices.records=[]; prices.checked.clear(); completed+=1
            if stop_after is not None and completed>=stop_after: break
        if checkpoint['next_day']==365:
            from export import finalize
            finalize(out,jan,b)
            b.check()
            save(out/'status.json',dict(status='PASS_ANNUAL_DRAFT_EXPORT',resources=b.record(),output_bytes=size_check(out,b),annual_complete=True))
        else: save(out/'status.json',dict(status='PAUSED_AT_DAY_BOUNDARY',next_day=checkpoint['next_day'],resources=b.record()))
        save(manifest,dict(identity=identity,run_status=read(out/'status.json')['status'],initial_energy_source={k:v for k,v in jan.items() if k!='ledger'},source_hashes=s.hashes))
    except BaseException:
        failure=dict(status='STOPPED_ERROR_OR_BUDGET',traceback=traceback.format_exc(),resources=b.record() if b else None)
        save(out/'failure.json',failure); save(out/'status.json',failure)
        if c is not None:
            try: pack(out/'failure_state.q4z',c.checkpoint())
            except Exception: pass
        raise
    finally:
        if b: b.persist()
        lock.unlink(missing_ok=True)

def preflight():
    ready=read(HERE/'outputs/ready.json'); sources=Sources(); data=old.load_inputs(sources); jan=january(data,sources); identity=source_identity(sources)
    assert ready['identity']['code']==identity['code'] and ready['identity']['model_sha256']==identity['model_sha256']
    for p,h in ready['source_hashes'].items(): assert sha(p)==h
    assert ready['within_planned_budget']
    print(json.dumps(dict(status='PASS_CONDA_ANNUAL_PREFLIGHT',conda_prefix=sys.prefix,initial_energy=jan['final_energy'],estimated_seconds=ready['estimated_seconds'],hard_limit_seconds=2400,solves=0,fits=0),ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['wiring','run','preflight']); p.add_argument('--out',type=Path,required=True); p.add_argument('--wiring',type=Path,default=HERE/'outputs/wiring_04'); p.add_argument('--resume',action='store_true'); p.add_argument('--user-run',action='store_true'); p.add_argument('--stop-after-days',type=int)
    a=p.parse_args()
    if a.mode=='wiring': wiring(a.out)
    elif a.mode=='preflight': preflight()
    else:
        if not a.user_run: p.error('Annual execution must be started by the user with --user-run')
        run(a.out,a.wiring,a.resume,a.stop_after_days)
