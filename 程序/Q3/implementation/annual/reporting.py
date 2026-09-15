"""Annual scientific answers and auditable DRAFT cell mappings (no formal XLSX)."""
import csv,io,os,math,time
from datetime import datetime,timedelta
from runtime import read,sha
from independent import audit_block,close

def csv_file(path,rows,budget):
    if not rows: return
    stream=io.StringIO(newline=''); w=csv.DictWriter(stream,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    text_file(path,stream.getvalue(),budget)
def text_file(path,text,budget):
    budget.check(); path.parent.mkdir(parents=True,exist_ok=True); old=path.stat().st_size if path.exists() else 0; tmp=path.with_name(path.name+'.tmp')
    if budget.bytes+len(text.encode('utf-8-sig'))-old>budget.limits['output_bytes']:
        from runtime import StopRun
        raise StopRun('Text output exceeds disk budget')
    with tmp.open('w',encoding='utf-8-sig',newline='') as f: f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path); budget.bytes+=path.stat().st_size-old; budget.check()
def column(n):
    s=''
    while n: n,r=divmod(n-1,26); s=chr(65+r)+s
    return s

def draft_maps(rows,out,budget,first_day=31,synthetic=False):
    """Delivery-key join; missing 2026 contract stays explicitly unresolved."""
    by={r['delivery_start']:r for r in rows}; maps=[]; reconciliation=[]; storage=[]; events=[]; membership=[]; micro=[]
    for d in range(first_day,365):
        stamp=datetime(2025,1,1)+timedelta(days=d); day=[by[(stamp+timedelta(minutes=10*s)).isoformat()] for s in range(144)]
        rr=d-29
        for sheet,field in [('计划购电量','q0'),('调整购电量','a_final')]:
            vals=[]; fees=[]
            for j in range(144):
                key=(stamp+timedelta(minutes=(j+1)*10)).isoformat(); r=by.get(key)
                value=None if r is None else r[field]; money=None if r is None else (r['price']*r['q0'] if field=='q0' else r['normal']+r['adjustment'])
                maps.append(dict(sheet=sheet,cell=f'{column(j+2)}{rr}',date=stamp.date().isoformat(),delivery_start=key,value=value,price=None if r is None else r['price'],fee=money,version_id=None if r is None else (f'C2_d{r["day"]}_s0' if field=='q0' else r['final_version_id']),published_position=None if r is None else (r['q0_created_at'] if field=='q0' else r['final_published_at']),status='UNRESOLVED_YEAR_END' if r is None else 'SOURCE_AVAILABLE'))
                vals.append(value); fees.append(money)
            complete=all(x is not None for x in vals)
            for col,value in [('EP',math.fsum(vals) if complete else None),('EQ',math.fsum(fees) if complete else None)]:
                maps.append(dict(sheet=sheet,cell=f'{col}{rr}',date=stamp.date().isoformat(),delivery_start='',value=value,price=None,fee=None,version_id='',published_position=None,status='SOURCE_AVAILABLE' if complete else 'INCOMPLETE_WINDOW'))
            fieldfees=[r['price']*r['q0'] if field=='q0' else r['normal']+r['adjustment'] for r in day]
            nextfirst=by.get((stamp+timedelta(days=1)).isoformat())
            diff=None if nextfirst is None else nextfirst[field]-day[0][field]
            if complete: close(math.fsum(vals),math.fsum(r[field] for r in day)+diff)
            reconciliation.append(dict(date=stamp.date().isoformat(),sheet=sheet,natural_quantity=math.fsum(r[field] for r in day),display_quantity=math.fsum(vals) if complete else None,quantity_difference=diff,natural_fee=math.fsum(fieldfees),display_fee=math.fsum(fees) if complete else None,status='PASS' if complete else 'UNKNOWN_YEAR_END'))
        for k in range(6):
            chunk=day[k*24:(k+1)*24]
            storage.append(dict(date=stamp.date().isoformat(),row=2+6*(d-31)+k,time_interval=f'{4*k}:00-{4*(k+1)}:00',charge=math.fsum(r['c'] for r in chunk),discharge=math.fsum(r['s'] for r in chunk),SOC_label='0:00' if k==0 else ('24:00' if k==1 else ''),SOC=day[0]['E_before'] if k==0 else (day[-1]['E_after'] if k==1 else None)))
        pending=[]; kind=None; chunks=[]
        for r in day:
            current='main' if r['g']>1e-6 else ('micro' if r['g']>0 else None)
            if current!=kind:
                if pending: chunks.append((kind,pending)); pending=[]
                kind=current
            if current: pending.append(r)
        if pending: chunks.append((kind,pending))
        if not chunks: events.append(dict(date=stamp.date().isoformat(),start='无',end='',kind='none',energy=0,fee=0))
        for kind,chunk in chunks:
            event=dict(date=stamp.date().isoformat(),start=chunk[0]['delivery_start'],end=chunk[-1]['delivery_end'],kind=kind,energy=math.fsum(r['g'] for r in chunk),fee=math.fsum(r['emergency'] for r in chunk))
            events.append(event)
            for r in chunk: membership.append(dict(delivery_start=r['delivery_start'],event_index=len(events),kind=kind,energy=r['g'],fee=r['emergency']))
        micro.append(dict(date=stamp.date().isoformat(),quantity=math.fsum(r['g'] for r in day if 0<r['g']<=1e-6),fee=math.fsum(r['emergency'] for r in day if 0<r['g']<=1e-6)))
    csv_file(out/'DRAFT/order_cell_mapping.csv',maps,budget); csv_file(out/'DRAFT/storage_mapping.csv',storage,budget); csv_file(out/'DRAFT/emergency_events.csv',events,budget); csv_file(out/'DRAFT/event_membership.csv',membership,budget); csv_file(out/'DRAFT/micro_emergency.csv',micro,budget); csv_file(out/'boundary_reconciliation.csv',reconciliation,budget)
    # Separate disk readback of generated mappings, joined through delivery keys.
    checked=0; missing=[]
    with (out/'DRAFT/order_cell_mapping.csv').open(encoding='utf-8-sig',newline='') as f:
        for item in csv.DictReader(f):
            if not item['delivery_start']: continue
            r=by.get(item['delivery_start'])
            if r is None:
                if item['value'] or item['cell']!='EO335': raise AssertionError('Invented draft boundary')
                missing.append(item['sheet']+'!'+item['cell']); continue
            close(float(item['value']),r['q0'] if item['sheet']=='计划购电量' else r['a_final']); checked+=1
    if checked!=2*((365-first_day)*144-1) or len(missing)!=2 or len(storage)!=6*(365-first_day): raise AssertionError('Draft coverage count')
    selected=[r for r in rows if r['day']>=31]
    close(math.fsum(e['energy'] for e in events),math.fsum(r['g'] for r in selected)); close(math.fsum(e['fee'] for e in events),math.fsum(r['emergency'] for r in selected),.01)
    decision=dict(order_owner_interpretation='delivery_archive',year_end_contract_rule='UNRESOLVED',extra_segment_control=False,forecast_fallback=None,affected_dates=['2025-12-31','2026-01-01'],changes_prior_horizon=False,user_decision_reference=None,formal_export_allowed=False,missing_cells=[s+'!'+c for s in ('计划购电量','调整购电量') for c in ('EO335','EP335','EQ335')],artifact_kind='DRAFT_MAPPING_CSV_NOT_WORKBOOK',known_order_cells=checked,synthetic=synthetic)
    budget.write(out/'boundary_decision.json',decision); budget.write(out/'DRAFT/mapping_readback.json',dict(status='PASS_KNOWN_CSV_VALUES',checked=checked,unresolved=missing,workbook_readback='NOT_RUN'))
    if '2025-02-01T00:00:00' in by: csv_file(out/'DRAFT/natural_Feb1_first_segment.csv',[by['2025-02-01T00:00:00']],budget)

def finish(out,raw,ident,budget):
    energies={p:6000. for p in ('C0','C1','C2')}; daily=[]; c2=[]; previous=None; counts={p:[0,0] for p in energies}
    selected_dates=('2025-03-20','2025-06-21','2025-09-23','2025-12-21')
    writers={}; files={}
    try:
        for policy in energies:
            path=out/policy/'slot_ledger.csv'; path.parent.mkdir(parents=True,exist_ok=True); files[policy]=(path,path.with_name(path.name+'.tmp').open('w',encoding='utf-8-sig',newline=''))
        for day in range(365):
            budget.check(); path=out/f'commits/{day:03d}.json'; cp=read(path)
            if cp['identity']!=ident or cp['next_day']!=day+1 or cp['previous_commit']!=previous: raise AssertionError('Broken commit chain')
            previous=sha(path)
            for policy in energies:
                rel=f'days/{day:03d}/{policy}.json.gz'
                if sha(out/rel)!=cp['files'][rel]: raise AssertionError('Changed committed block')
                block=read(out/rel); result=audit_block(block,raw,energies[policy],day*144,(day+1)*144,out/'cache')
                for key in ('normal','adjustment','emergency','total','c','s','g','w','initial_SOC','final_SOC'): close(result[key],cp['audits'][policy][key],1e-4)
                energies[policy]=result['final_SOC']; counts[policy][0]+=result['segments']; counts[policy][1]+=result['plans']; result['date']=(datetime(2025,1,1)+timedelta(days=day)).date().isoformat(); daily.append(result)
                if policy not in writers:
                    writers[policy]=csv.DictWriter(files[policy][1],fieldnames=list(block['ledger'][0])); writers[policy].writeheader()
                writers[policy].writerows(block['ledger'])
                if policy=='C2':
                    c2.extend(block['ledger'])
                    if result['date'] in selected_dates:
                        csv_file(out/f'paper_tables/{result["date"]}_ten_minute.csv',block['ledger'],budget)
                        budget.write(out/f'paper_tables/{result["date"]}_order_versions.json',block['versions'])
        head=read(out/'checkpoint.json')
        if head['commit_hash']!=previous: raise AssertionError('Final head not last annual commit')
        for policy in energies:
            if counts[policy]!=[52560,8754]: raise AssertionError('Annual segment/plan count')
            path,f=files[policy]; f.flush(); os.fsync(f.fileno()); f.close(); old=path.stat().st_size if path.exists() else 0; tmp=path.with_name(path.name+'.tmp'); os.replace(tmp,path); budget.bytes+=path.stat().st_size-old
    finally:
        for path,f in files.values():
            if not f.closed: f.close()
    monthly=[]; annual={}; fields=('normal','adjustment','emergency','total','g','w','c','s')
    for policy in energies:
        dd=[r for r in daily if r['policy']==policy]
        for month in range(1,13):
            selected=[r for r in dd if int(r['date'][5:7])==month]
            monthly.append(dict(policy=policy,month=month,days=len(selected),**{k:math.fsum(r[k] for r in selected) for k in fields}))
        annual[policy]=dict(J_Feb_Dec=math.fsum(r['total'] for r in dd[31:]),January=math.fsum(r['total'] for r in dd[:31]),full_year=math.fsum(r['total'] for r in dd),Feb1_SOC=dd[31]['initial_SOC'],final_SOC=energies[policy],segments=52560,main_segments=48096,plans=8754,main_components={k:math.fsum(r[k] for r in dd[31:]) for k in fields})
        close(annual[policy]['full_year'],math.fsum(r['total'] for r in monthly if r['policy']==policy),.01)
    delta_info=annual['C0']['J_Feb_Dec']-annual['C1']['J_Feb_Dec']; delta_adjust=annual['C1']['J_Feb_Dec']-annual['C2']['J_Feb_Dec']; delta_total=annual['C0']['J_Feb_Dec']-annual['C2']['J_Feb_Dec']; close(delta_total,delta_info+delta_adjust,.01)
    answer=dict(branches=annual,delta_info=delta_info,delta_adjust=delta_adjust,delta_total=delta_total,numerical_tie_yuan=.01,scope='2025 Feb-Dec; complete policy paths; no isolated causal or universal optimality claim',C2_is_main_output=True)
    csv_file(out/'daily.csv',daily,budget); csv_file(out/'monthly.csv',monthly,budget); budget.write(out/'annual_summary.json',answer); budget.write(out/'independent_audit/summary.json',dict(status='PASS',counts=counts,main_segments_each=48096,all_days_reloaded=True,all_commits_hashed=True))
    draft_maps(c2,out,budget)
    text_file(out/'necessity_verdict.md',f'# Q3年度政策比较\n\n主评价为2025年2—12月自然日真实账本。\n\nΔ_info={delta_info:.10f}元；Δ_adjust={delta_adjust:.10f}元；Δ_total={delta_total:.10f}元。正值表示前一分支费用更高，负值原样保留，绝对值≤0.01元记数值持平。三条策略具有各自的SOC历史，不能解释为同状态的单次信息因果效应。C2继续作为题目主输出。\n\n年度逐日重新读回审计通过。DRAFT目录为有来源单元格映射CSV，并非已保存读回的四表工作簿；2026首段合同未决，正式result3.xlsx禁止生成。\n',budget)
    return answer
