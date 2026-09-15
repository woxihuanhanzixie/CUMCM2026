"""Separate read-only replay of every actual and every raw/restored remaining LP.
Does not call the controller's builders, restore function, or common ledger.
"""
import runtime
from runtime import ROOT
import json
from pathlib import Path
from datetime import timedelta
import numpy as np
from workflow import load_data,save,identity,digest_array
from v2_impl.common.data_io import sha256_file

def run():
    out=ROOT/'outputs/M1_D48';data=load_data()
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    assert identity()['signature']==manifest['signature'],'source/input/config/environment changed'
    jan=json.loads((out/'january_summary.json').read_text(encoding='utf-8'))
    e=jan['feb1_energy'];errors=[];worst={'balance':0.,'state':0.,'physical':0.,'bill':0.,'lp_balance':0.,'lp_state':0.,'lp_bounds':0.,'lp_modes':0.,'restore_objective_increase':0.}
    totals=np.zeros(2);calls=0;fallbacks=0;success=0;feasible=0;days=[]
    solver_calls=0;secondary_accepted=0;secondary_fallbacks=0;secondary_statuses={}
    worst.update(root_rule_kwh=0.,secondary_cost_excess_yuan=0.,secondary_energy_loss_kwh=0.)
    tariffs=dict(zip(data['labels'],data['price']))
    for index,path in enumerate(sorted((out/'days').glob('*.json'))):
        r=json.loads(path.read_text(encoding='utf-8'));d=31+index
        assert r['day']==d and r['date']==data['dates'][d].isoformat()
        assert abs(r['e0']-e)<=1e-6
        ar={k:np.array(r[k]) for k in ('q','c','s','z','w','E')}
        assert all(a.shape==(144,) and np.isfinite(a).all() for a in ar.values())
        q,c,s,z,w,E=(ar[k] for k in ('q','c','s','z','w','E'))
        n=(data['load_act'][d]-data['pv_act'][d])/6
        worst['balance']=max(worst['balance'],float(abs(q-c+s+z-w-n).max()))
        worst['state']=max(worst['state'],float(abs(E-np.r_[e,E[:-1]]-.9*c+s/.9).max()))
        worst['physical']=max(worst['physical'],float(-min(q.min(),c.min(),s.min(),z.min(),w.min())),float(1200-E.min()),float(E.max()-10800),float(max(c.max(),s.max())-5000/6),float(np.minimum(c,s).max()),float(np.minimum(c,z).max()),float(np.minimum(z,w).max()))
        costs=np.array([sum(data['price'][t]*q[t] for t in range(144)),sum(5*data['price'][t]*z[t] for t in range(144))])
        worst['bill']=max(worst['bill'],float(abs(costs-[r['plan_cost'],r['emg_cost']]).max()),abs(float(costs.sum())-r['total_cost']))
        totals+=costs
        lock=json.loads((out/'locks'/f'{d:03d}.json').read_text(encoding='utf-8'))
        assert lock['event']=='locked_before_first_observation'
        np.testing.assert_array_equal(q,lock['q']);assert digest_array(q)==r['lock_sha256']==lock['sha256']
        fc=json.loads((out/'forecasts'/f'{d:03d}.json').read_text(encoding='utf-8'))
        assert set(fc['predictions'])==({'0'} if d==364 else {'0','1'})
        for ref in fc['refs']:
            k=ref['k'];assert ref['publish_day']==d and ref['target_day']==d+k and k in (0,1)
            assert ref['training_target_end_inclusive']<d
            assert ref['n_train']==min(28,d-k-14)
        assert sha256_file(out/'lp'/f'{d:03d}.npz')==r['lp_archive_sha256']
        with np.load(out/'lp'/f'{d:03d}.npz') as arrays:
            for slot,record in [(-1,r['midnight'])]+list(enumerate(r['intraday'])):
                calls+=1;status=record.get('status',record.get('lp_status'))
                solver_calls+=record.get('solver_calls',1)
                success+=status==0;fallbacks+=record['fallback'];feasible+=not record['fallback']
                prefix='midnight_' if slot==-1 else f't{slot:03d}_'
                t=max(slot,0);h=144*(1 if d==364 else 2)-t
                assert record['H']==h
                expected=[]
                for k in range(1 if d==364 else 2):
                    value=(np.array(fc['predictions'][str(k)]['load'])-np.array(fc['predictions'][str(k)]['pv']))/6
                    expected.extend(value[t:] if k==0 else value)
                if slot!=-1:expected[0]=n[t]
                np.testing.assert_allclose(arrays[prefix+'nbar'],expected,atol=1e-9,rtol=0)
                price=[]
                for u in range(h):
                    endpoint=data['dates'][d]+timedelta(minutes=(t+u+1)*10)
                    label='0:00+1' if endpoint.hour==endpoint.minute==0 else f'{endpoint.hour}:{endpoint.minute:02d}'
                    price.append(tariffs[label])
                np.testing.assert_array_equal(arrays[prefix+'pbar'],price)
                if record['fallback']:continue
                nF=0 if slot==-1 else 144-t
                Q=arrays[prefix+'raw_lp_Q'];initial=e if slot==-1 else (e if t==0 else E[t-1])
                objs=[]
                for mode in (('raw_lp_','restored_lp_') if slot==-1 else ('raw_lp_','restored_lp_','primary_lp_')):
                    Q=arrays[prefix+('primary_lp_Q' if mode=='primary_lp_' else 'raw_lp_Q')]
                    cs,ss,ws,es=(arrays[prefix+mode+k] for k in ('c','s','w','E'))
                    zs=np.zeros(h) if slot==-1 and mode=='raw_lp_' else arrays[prefix+mode+'z']
                    worst['lp_balance']=max(worst['lp_balance'],float(abs(Q-cs+ss+zs-ws-expected).max()))
                    worst['lp_state']=max(worst['lp_state'],float(abs(es-np.r_[initial,es[:-1]]-.9*cs+ss/.9).max()))
                    bound=max(0.,float(-min(Q.min(),cs.min(),ss.min(),zs.min(),ws.min())),float(max(cs.max(),ss.max())-5000/6),float(1200-es.min()),float(es.max()-10800))
                    if nF:
                        bound=max(bound,float(abs(Q[:nF]-q[t:]).max()),float((cs[:nF]-np.maximum(q[t:]-np.array(expected[:nF]),0)).max()))
                        cstar=min(max(q[t]-n[t],0.),5000/6,max(10800-initial,0.)/.9)
                        scap=min(max(n[t]-q[t],0.),5000/6,max(initial-1200,0.)*.9)
                        worst['root_rule_kwh']=max(worst['root_rule_kwh'],abs(float(cs[0])-cstar),float(ss[0])-scap)
                    if nF<h:bound=max(bound,float(abs(zs[nF:]).max()))
                    worst['lp_bounds']=max(worst['lp_bounds'],bound)
                    if mode=='restored_lp_':
                        worst['lp_modes']=max(worst['lp_modes'],float(np.minimum(cs,ss).max()),float(np.minimum(zs,ws).max()),float(np.minimum(cs,zs).max()))
                        if slot!=-1:
                            for name,v in zip(('c','s','z','w','E'),(cs,ss,zs,ws,es)):assert abs(ar[name][t]-v[0])<=1e-6
                    objs.append(5*np.dot(price[:nF],zs[:nF])+np.dot(price[nF:],Q[nF:]))
                worst['restore_objective_increase']=max(worst['restore_objective_increase'],float(objs[1]-objs[0]))
                if slot!=-1:
                    assert abs(objs[2]-record['primary_fun'])<1e-6
                    sec=record['secondary'];secondary_accepted+=sec['accepted']
                    sk=str(sec.get('status','not_returned'))
                    secondary_statuses[sk]=secondary_statuses.get(sk,0)+1
                    secondary_fallbacks+=not sec['accepted']
                    worst['secondary_cost_excess_yuan']=max(worst['secondary_cost_excess_yuan'],float(objs[0]-objs[2]-1e-6))
                    worst['secondary_energy_loss_kwh']=max(worst['secondary_energy_loss_kwh'],float(arrays[prefix+'primary_lp_E'][0]-arrays[prefix+'raw_lp_E'][0]))
        e=E[-1];days.append(d)
    assert len(days)==summary['complete_days'] and calls==145*len(days)
    assert solver_calls==summary['lp_calls'] and calls==summary['decision_count']
    assert secondary_accepted==summary['secondary_accepted'] and secondary_fallbacks==summary['secondary_fallback_count']
    worst['bill']=max(worst['bill'],float(abs(totals-[summary['plan_cost'],summary['emergency_cost']]).max()))
    for k,v in worst.items():
        if v>(.01 if k=='bill' else 1e-6):errors.append(f'{k}: {v}')
    report={'passed':not errors,'scope':f'{len(days)} days actual/locked orders and {calls} raw/restored LPs; input chronology; '+('annual' if len(days)==334 else 'NOT annual'),
        'days':days,'slots':len(days)*144,'decision_count':calls,'lp_calls':solver_calls,
        'secondary_accepted':secondary_accepted,'secondary_fallbacks':secondary_fallbacks,
        'secondary_status_counts':secondary_statuses,
        'solver_success_status0':int(success),'feasible_solutions':int(feasible),'fallback_count':int(fallbacks),
        'maxima':worst,'plan_cost':totals[0],'emergency_cost':totals[1],'total_cost':totals.sum(),'end_energy':e,'errors':errors,
        'audit_source_sha256':sha256_file(Path(__file__)), 'run_signature':manifest['signature']}
    save(ROOT/'evidence/independent_run_audit.json',report)
    summary.update(independent_audit_passed=report['passed'],independent_audit_scope=report['scope'])
    save(out/'summary.json',summary)
    print(json.dumps(report,ensure_ascii=False));assert report['passed']
    return report
if __name__=='__main__':run()
