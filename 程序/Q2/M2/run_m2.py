"""M2 only: preflight, bounded rolling run/resume, audit, CT4 diagnostics."""
from __future__ import annotations
import runtime  # noqa: F401
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
import scipy
import openpyxl
from scipy.optimize import milp,Bounds,LinearConstraint
from controller import Controller,ForecastArchive,digest,forecast_at
from audit import audit_records
from v2_impl.common import data_io,template_map
from v2_impl.common.january import run_january,check_january
from v2_impl.common.result_writer import write_result2,check_result2

ROOT=Path(__file__).resolve().parent


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
                               default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),encoding='utf-8')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_data():
    if (data_io.sha256_file(data_io.A1_DEFAULT)!=data_io.REF_HASH_A1 or
            data_io.sha256_file(data_io.A2_DEFAULT)!=data_io.REF_HASH_A2):
        raise ValueError('original attachment hashes differ from frozen Q2 inputs')
    a1=data_io.load_attachment1();a2=data_io.load_attachment2()
    if (a2['load_act'].shape!=(365,144) or a2['pv_act'].shape!=(365,144)
            or not np.isfinite(a2['load_act']).all() or not np.isfinite(a2['pv_act']).all()
            or not np.isfinite(a1['price']).all() or (a1['price']<=0).any()):
        raise ValueError('invalid raw data dimensions/values')
    return {**a2,'price':a1['price'],'load_a1':a1['load'],'pv_a1':a1['pv']}


def signature(args):
    code={str(p.relative_to(ROOT)):data_io.sha256_file(p) for p in sorted(ROOT.glob('*.py'))}
    code.update({str(p.relative_to(ROOT)):data_io.sha256_file(p) for p in sorted((ROOT/'v2_impl').rglob('*.py'))})
    return dict(model='M2_H48CT3',horizon_hours=48,terminal_value=0,
                source_definition='existing m2.py + spec04; user confirmed 2026-09-12',
                python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                openpyxl=openpyxl.__version__,platform=platform.platform(),code_hashes=code,
                inputs={str(p):data_io.sha256_file(p) for p in
                        (data_io.A1_DEFAULT,data_io.A2_DEFAULT,data_io.TEMPLATE_DEFAULT)},
                time_limit=args.time_limit,gap=args.gap,formulation=args.formulation)


def preflight(args):
    start=time.perf_counter();data=load_data()
    jan=run_january(data['price'],data['load_act'],data['pv_act'])
    pr=forecast_at(data['load_act'][:31],data['pv_act'][:31],data['dates'],31)
    r=milp([-1.],integrality=[1],bounds=Bounds([0],[3]),constraints=LinearConstraint([[1]],[-np.inf],[2.5]))
    checks=dict(a1_hash=data_io.sha256_file(data_io.A1_DEFAULT)==data_io.REF_HASH_A1,
                a2_hash=data_io.sha256_file(data_io.A2_DEFAULT)==data_io.REF_HASH_A2,
                shapes=data['load_act'].shape==data['pv_act'].shape==(365,144),
                finite=bool(np.isfinite(data['load_act']).all() and np.isfinite(data['pv_act']).all()),
                pv_nonnegative=bool((data['pv_act']>=0).all()),
                headers=not data_io.check_attachment2_header(data['load_header'],data['pv_header']),
                template=not template_map.check_template(data_io.TEMPLATE_DEFAULT),
                january=not check_january(jan),forecast=pr.shape==(2,3,144),
                milp=bool(r.success and abs(r.x[0]-2)<1e-8))
    report=dict(passed=all(checks.values()),checks=checks,environment=signature(args),
                executable=sys.executable,january_cost=jan['jan_cost'],feb1_energy=jan['feb1_energy'],
                seconds=time.perf_counter()-start,annual_started=False)
    atomic_json(ROOT/'evidence/preflight.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='environment'},ensure_ascii=False),flush=True)
    return 0 if report['passed'] else 1


def completed(out):
    return [r for p in sorted((out/'days').glob('day_*.json')) if (r:=read(p))['next_slot']==144]


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def export(out,data,jan):
    recs=completed(out);result=audit_records(recs,data,jan['feb1_energy'])
    result['january_cost']=jan['jan_cost'];result['january_plus_scored_cost']=jan['jan_cost']+result['total_cost']
    logs=[r for rec in recs for r in rec['logs']]
    times=np.array([r.get('total_seconds',r['build_seconds']+r['solve_seconds']) for r in logs])
    if len(times):
        result['timing']=dict(calls=len(times),average=float(times.mean()),p95=float(np.quantile(times,.95)),
                              maximum=float(times.max()),optimizer_hours_48430=float(times.mean()*48430/3600))
    if recs:
        arrays={k:np.array([rec[k] for rec in recs]) for k in ('q','c','s','z','w','E')}
        dates=[data['dates'][r['day']] for r in recs]
        book=out/('result2_candidate.xlsx' if result['annual_complete'] else 'result2_partial.xlsx')
        write_result2(book,dates,arrays['q'],arrays['c'],arrays['s'],arrays['z'],arrays['E'],data['price'])
        errors=check_result2(book,dates,arrays['q'],arrays['c'],arrays['s'],arrays['z'],arrays['E'],data['price'])
        result['workbook_errors']=errors
        if errors:result['independent_audit_passed']=False
        rows=[]
        for rec in recs:
            d=rec['day'];date=data['dates'][d].date().isoformat()
            for t in range(144):
                rows.append(dict(day=d,date=date,slot=t,**{k:rec[k][t] for k in arrays}))
        write_csv(out/'trajectory.csv',rows)
        daily=[]
        for r,a in zip(recs,result['days']):
            d=r['day'];daily.append(dict(day=d,date=data['dates'][d].date().isoformat(),
                                        plan_cost=a['plan_cost'],emergency_cost=a['emergency_cost'],total=a['total'],
                                        initial_energy=r['e0'],end_energy=a['end_energy'],
                                        plan_kwh=sum(r['q']),emergency_kwh=sum(r['z']),
                                        charge_kwh=sum(r['c']),discharge_kwh=sum(r['s']),unused_kwh=sum(r['w'])))
        write_csv(out/'daily_summary.csv',daily)
        months={}
        for row in daily:
            month=row['date'][:7];m=months.setdefault(month,dict(month=month,days=0,plan_cost=0.,emergency_cost=0.,total=0.))
            m['days']+=1
            for k in ('plan_cost','emergency_cost','total'):m[k]+=row[k]
        write_csv(out/'monthly_summary.csv',list(months.values()))
        write_csv(out/'midnight_orders.csv',[{k:r[k] for k in ('day','date','slot','q')} for r in rows])
        fields=sorted(set().union(*(r.keys() for r in logs)))
        with (out/'solver_log.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fields);w.writeheader();w.writerows(logs)
        atomic_json(out/'fallback_log.json',[r for r in logs if r['fallback']])
    atomic_json(out/'independent_audit.json',result)
    return result


def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=out/'run.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError(f'Run locked: {lock}; verify previous process is stopped before removing stale lock')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:return _run(args,out)
    finally:lock.unlink(missing_ok=True)


def _run(args,out):
    sig=signature(args);manifest=out/'manifest.json'
    if manifest.exists():
        if not args.resume:raise ValueError('output exists; use --resume or a new --output')
        if read(manifest)!=sig:raise ValueError('code/input/environment/config changed; use new output, never reuse these checkpoints')
    else:atomic_json(manifest,sig)
    data=load_data();jan=run_january(data['price'],data['load_act'],data['pv_act'])
    if check_january(jan):raise ValueError('January reference failed')
    archive=ForecastArchive(out/'forecasts',data['dates'])
    ctrl=Controller(data['price'],args.time_limit,args.gap,args.formulation)
    start=time.perf_counter();calls=0;elapsed_optimizer=0.;energy=jan['feb1_energy'];paused=False
    for d in range(31,31+args.days):
        path=out/'days'/f'day_{d:03d}.json'
        if path.exists():
            rec=read(path)
            if abs(rec['e0']-energy)>1e-6:raise ValueError('checkpoint cross-day state mismatch')
            if rec['next_slot']==144:
                energy=rec['E'][-1];continue
        else:
            snap=archive.snapshot(d,data['load_act'][:d],data['pv_act'][:d])
            q,log=ctrl.midnight(snap,energy);calls+=1
            elapsed_optimizer+=log.get('total_seconds',log['build_seconds']+log['solve_seconds'])
            rec=dict(day=d,e0=float(energy),next_slot=0,q=q.tolist(),q_hash=hashlib.sha256(q.tobytes()).hexdigest(),
                     snapshot=snap,snapshot_hash=digest(snap),logs=[log],
                     **{k:[0.]*144 for k in ('c','s','z','w','E')})
            atomic_json(path,rec)
        q=np.asarray(rec['q']);snap=rec['snapshot']
        energy=rec['e0'] if rec['next_slot']==0 else rec['E'][rec['next_slot']-1]
        for t in range(rec['next_slot'],144):
            if time.perf_counter()-start>args.wall_hours*3600 or (args.max_calls and calls>=args.max_calls):
                paused=True;break
            observed=float((data['load_act'][d,t]-data['pv_act'][d,t])/6)
            action,log=ctrl.step(snap,energy,q,t,observed);energy=action['E']
            for k in ('c','s','z','w','E'):rec[k][t]=action[k]
            rec['logs'].append(log);rec['next_slot']=t+1;calls+=1
            elapsed_optimizer+=log.get('total_seconds',log['build_seconds']+log['solve_seconds'])
            if (t+1)%12==0 or t==143:
                atomic_json(path,rec)
                print(f'{data["dates"][d].date()} slot {t+1}/144 E={energy:.3f} '
                      f'calls={calls} elapsed={time.perf_counter()-start:.1f}s '
                      f'fallbacks={sum(r["fallback"] for r in rec["logs"])}',flush=True)
        atomic_json(path,rec)
        if paused:break
        from audit import audit_day
        check=audit_day(rec,data['load_act'][d],data['pv_act'][d],data['price'],rec['e0'])
        if not check['passed']:raise ValueError(f'day {d} independent audit failed: {check}')
    report=export(out,data,jan)
    report['invocation']=dict(paused=paused,calls=calls,wall_seconds=time.perf_counter()-start,
                              optimizer_seconds=elapsed_optimizer,requested_days=args.days)
    atomic_json(out/'run_status.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('days',)},ensure_ascii=False),flush=True)
    return 0 if (report['independent_audit_passed'] or paused) else 1


def ct4(args):
    out=args.output.resolve();data=load_data();rows=[]
    ctrl=Controller(data['price'],60.,args.gap,args.formulation)
    for day in (31,78,171,265,354):
        path=out/'days'/f'day_{day:03d}.json'
        if not path.exists():rows.append(dict(day=day,status='missing frozen midnight state'));continue
        rec=read(path);res,_=ctrl.optimize(rec['snapshot'],rec['e0'],ct4=True)
        baseline=rec['logs'][0]
        if not res.usable or baseline['fallback']:
            rows.append(dict(day=day,status='unable_to_judge',solver=res.log));continue
        q4=np.array([res.orders[(('today',),u)] for u in range(144)])
        q3=np.asarray(rec['q']);J3=baseline['restored_objective'];J4=res.log['restored_objective']
        dq=float(np.abs(q4-q3).sum()/max(1.,abs(q3).sum()));dj=abs(J4-J3)/max(1.,abs(J3))
        reliable=baseline['solver_success'] and res.log['solver_success']
        rows.append(dict(day=day,status='diagnostic' if reliable else 'unable_to_judge_gap',
                         q_relative=dq,objective_relative=dj,ct3_sensitive=(dq>.05 or dj>.01) if reliable else None,
                         solver=res.log))
    atomic_json(out/'ct4_diagnostic.json',rows);print(json.dumps(rows,ensure_ascii=False));return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('preflight','run','audit','ct4'))
    p.add_argument('--output',type=Path,default=ROOT/'outputs/M2_H48CT3')
    p.add_argument('--days',type=int,default=2,help='total scored days beginning Feb1, including completed days')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--time-limit',type=float,default=5.)
    p.add_argument('--gap',type=float,default=1e-4)
    p.add_argument('--formulation',choices=('original','reduced'),default='reduced')
    p.add_argument('--wall-hours',type=float,default=12.)
    p.add_argument('--max-calls',type=int,default=0,help='bounded smoke run; 0 means no call cap')
    args=p.parse_args()
    if not 1<=args.days<=334 or args.time_limit<=0 or args.gap<0 or args.wall_hours<=0 or args.max_calls<0:
        p.error('invalid duration/solver limits')
    if args.command=='preflight':return preflight(args)
    if args.command=='run':return run(args)
    if args.command=='ct4':return ct4(args)
    data=load_data();jan=run_january(data['price'],data['load_act'],data['pv_act'])
    report=export(args.output.resolve(),data,jan)
    print(json.dumps({k:v for k,v in report.items() if k!='days'},ensure_ascii=False))
    return 0 if report['independent_audit_passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
