"""Independent raw-attachment replay, with no controller/physics helper calls."""
import runtime  # noqa: F401
import json
from pathlib import Path
import numpy as np


def audit_day(record,load,pv,price,expected_energy):
    q,c,s,z,w,E=[np.asarray(record[k],float) for k in ('q','c','s','z','w','E')]
    checks={}
    checks['finite']=all(a.shape==(144,) and np.isfinite(a).all() for a in (q,c,s,z,w,E))
    if not checks['finite']:
        return dict(passed=False,checks=checks)
    net=(np.asarray(load)-np.asarray(pv))/6
    before=np.r_[expected_energy,E[:-1]]
    balance=float(np.max(abs(q-c+s+z-w-net)))
    state=float(np.max(abs(E-before-.9*c+s/.9)))
    checks.update(balance=balance<=1e-6,state=state<=1e-6,
                  initial=abs(record['e0']-expected_energy)<=1e-6,
                  nonnegative=min(a.min() for a in (q,c,s,z,w))>=-1e-6,
                  storage=E.min()>=1200-1e-6 and E.max()<=10800+1e-6,
                  port=max(c.max(),s.max())<=5000/6+1e-6,
                  modes=not bool(np.any((c>1e-6)&((s>1e-6)|(z>1e-6)))),
                  emergency_waste=not bool(np.any((z>1e-6)&(w>1e-6))),
                  emergency_cap=bool(np.all(z<=np.maximum(net,0)+1e-6)))
    plan=float(np.dot(price,q));emergency=float(5*np.dot(price,z))
    return dict(passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},
                balance_residual=balance,state_residual=state,plan_cost=plan,
                emergency_cost=emergency,total=plan+emergency,end_energy=float(E[-1]))


def audit_records(records,data,initial=10800.):
    energy=initial;days=[];errors=[]
    for expected_day,rec in enumerate(records,start=31):
        d=rec['day']
        if d!=expected_day or rec.get('next_slot')!=144:
            errors.append(f'noncontiguous/incomplete day {d}')
        day=audit_day(rec,data['load_act'][d],data['pv_act'][d],data['price'],energy)
        day['day']=d;days.append(day)
        if not day['passed']:
            errors.append(f'physics failed day {d}')
        q_hash=rec.get('q_hash')
        import hashlib
        if q_hash!=hashlib.sha256(np.asarray(rec['q'],float).tobytes()).hexdigest():
            errors.append(f'locked order changed day {d}')
        logs=rec['logs']
        if len(logs)!=145 or logs[0]['kind']!='midnight' or [r['slot'] for r in logs[1:]]!=list(range(144)):
            errors.append(f'incomplete optimizer logs day {d}')
        for log in logs:
            if log['day']!=d or log['snapshot_hash']!=rec['snapshot_hash']:
                errors.append(f'wrong daily snapshot day {d}')
        energy=float(rec['E'][-1])
    logs=[r for rec in records for r in rec['logs']]
    complete=len(records)==334 and not errors
    return dict(independent_audit_passed=bool(records) and not errors,annual_complete=complete,
                candidate_solved=complete,user_adopted=False,days=days,errors=errors,
                total_cost=sum(d['total'] for d in days),plan_cost=sum(d['plan_cost'] for d in days),
                emergency_cost=sum(d['emergency_cost'] for d in days),end_energy=energy,
                fallback_count=sum(bool(r['fallback']) for r in logs),
                optimizer_fully_met=bool(logs) and all(r['solver_success'] and not r['fallback'] for r in logs))
