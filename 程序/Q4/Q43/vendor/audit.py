"""Scalar/raw-source checks independent of LP elimination and execution function."""
from datetime import datetime,timedelta
import math, copy
import numpy as np
from openpyxl import load_workbook
from core import P,ETA,LO,HI

def physical(E,ell,pv,A,act):
    c,s,g,w,En=[float(act[k]) for k in ('c','s','g','w','E')]
    residual=max(abs(A+pv+s+g-ell-c-w),abs(En-E-.9*c+s/.9),max(0,1200-En,En-10800,c-5000/6,s-5000/6,-min(A,c,s,g,w)))
    if residual>1e-6: raise AssertionError(('physical residual',residual))
    if any(x>1e-6 and y>1e-6 for x,y in ((c,s),(c,g),(s,w),(g,w))): raise AssertionError('Physical mutual exclusion')
    return residual

def raw_audit_sources(root):
    wb=load_workbook(root/'C题/附件/附件2.xlsx',read_only=True,data_only=True); raw={}
    for sheet,k in [('小区负载','ell'),('光伏发电实际功率','pv')]:
        # Read only the original cells necessary to audit the executed two days.
        rows=list(wb[sheet].iter_rows(min_row=2,max_row=3,min_col=2,max_col=145,values_only=True)); raw[k]=[float(v)/6 for row in rows for v in row]
    wb.close(); wb=load_workbook(root/'C题/附件/附件1.xlsx',read_only=True,data_only=True); rows=list(wb.active.values); pc=list(rows[0]).index('电价')
    raw['prices']={((int(str(r[0]).split(':')[0])%24)*60+int(str(r[0]).split(':')[1][:2])):float(r[pc]) for r in rows[1:]}; wb.close(); return raw

def audit_controller(controller,raw):
    rows=controller.ledger; plans=controller.plans; policy=controller.access.policy
    if len(rows)!=288 or [r['position'] for r in rows]!=list(range(288)): raise AssertionError('Missing or duplicate deliveries')
    if len(plans)!=42 or [p['position'] for p in plans]!=list(range(36,288,6)): raise AssertionError('Wrong optimization clock')
    maxres=0.; maxbill=0.; actualE=6000.; sums={'normal':0.,'adjustment':0.,'emergency':0.,'total':0.,'w':0.}; perday={}
    versions=controller.versions; planmap={p['position']:p for p in plans}
    expect=[0,144] if policy!='C2' else [0,36,72,108,144,180,216,252]
    if [v['published_position'] for v in versions]!=expect: raise AssertionError('Missing legal order version')
    for r in rows:
        i=r['position']; d,t=divmod(i,144); node=max(i-1,0); stamp=datetime(2025,1,1)+timedelta(minutes=i*10)
        if r['delivery_start']!=stamp.isoformat() or r['delivery_end']!=(stamp+timedelta(minutes=10)).isoformat() or r['source_node_index']!=node or r['initial_missing_node_approximation']!=(i==0): raise AssertionError('Raw time mapping')
        L,V=raw['ell'][node],raw['pv'][node]; price=raw['prices'][t*10]
        if max(abs(r['ell']-L),abs(r['pv']-V),abs(r['price']-price),abs(r['E_before']-actualE))>1e-6: raise AssertionError('Raw source or state continuity')
        day_versions=[v for v in versions if v['day']==d]; original=day_versions[0]; eligible=[v for v in day_versions if v['published_position']<=i and v['start']<=i<v['end']]; latest=eligible[-1]
        A=float(latest['orders'][i-latest['start']]); q=float(original['orders'][t])
        if abs(A-r['a_final'])>1e-6 or abs(q-r['q0'])>1e-6 or r['final_version_id']!=latest['id'] or r['final_published_at']!=latest['published_position']: raise AssertionError('Contract version join')
        if policy!='C2' and A!=q: raise AssertionError('Locked policy changed orders')
        if any(not np.array_equal(v['q0'],original['q0']) for v in day_versions): raise AssertionError('Original contract rewritten')
        R=1200. if i<36 else planmap[i-i%6]['solution']['E'][i%6]
        if abs(R-r['R'])>1e-6 or (i>=36 and r['hour_plan_time']!=i-i%6): raise AssertionError('Non-native reference')
        act={k:r[k] for k in ('c','s','g','w')}; act['E']=r['E_after']; maxres=max(maxres,physical(actualE,L,V,A,act)); actualE=r['E_after']
        normal=price*A; adjust=.5*price*abs(A-q); emergency=5*price*r['g']; total=normal+adjust+emergency
        maxbill=max(maxbill,abs(r['total']-total),abs(r['normal']-normal),abs(r['adjustment']-adjust),abs(r['emergency']-emergency))
        if maxbill>1e-6: raise AssertionError('Segment bill')
        for k,value in [('normal',normal),('adjustment',adjust),('emergency',emergency),('total',total),('w',r['w'])]: sums[k]+=value
        perday.setdefault(d,[]).append(total)
    for d,values in perday.items():
        if abs(math.fsum(values)-math.fsum(r['total'] for r in rows if r['day']==d))>1e-4: raise AssertionError('Daily bill')
    if abs(sums['total']-math.fsum(r['total'] for r in rows))>1e-3: raise AssertionError('Whole small-run bill')
    log=controller.access.log
    for i in range(288):
        events=[e for e in log if e['position']==i]; kinds=[e['kind'] for e in events]
        if kinds.count('commit')!=1 or kinds.count('reveal')!=1 or kinds.index('commit')>kinds.index('reveal'): raise AssertionError('Commit-before-reveal')
        for e in events:
            if e['kind']=='completed' and e['end']>i: raise AssertionError('Future completed observation')
            if e['kind']=='history' and e['day']*144>i: raise AssertionError('Future training')
            if e['kind']=='official' and (e['day']*144+e['slot']>i or (policy=='C0' and e['slot']!=0)): raise AssertionError('Official release permission')
    for plan in plans:
        p=plan['problem']; sol=plan['solution']; i=plan['position']; d,t=divmod(i,144)
        if len(p['ell'])!=min((d+2)*144,365*144)-i or len(sol['E'])!=len(p['ell']): raise AssertionError('Planning resolution/domain')
        if abs(p['energy']-rows[i]['E_before'])>1e-6: raise AssertionError('Planning state reset')
        if plan['stage']=='feedback' and np.max(abs(sol['A'][:144-t]-p['current']))>1e-6: raise AssertionError('Feedback changed contracts')
    return dict(status='PASS',segments=288,plans=42,normal=sums['normal'],adjustment=sums['adjustment'],emergency=sums['emergency'],total=sums['total'],discarded_kwh=sums['w'],final_SOC=actualE,max_physical_residual=maxres,max_segment_bill_error=maxbill,max_SOC_reference_difference=max(abs(r['E_after']-r['R']) for r in rows[36:]),official_reads=sum(e['kind']=='official' for e in log))
