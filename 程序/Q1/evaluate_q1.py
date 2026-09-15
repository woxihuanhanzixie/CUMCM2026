#!/usr/bin/env python3
"""Q1 evaluation and sensitivity analysis."""

from __future__ import annotations

import argparse, csv, json, math, time
from pathlib import Path
from typing import Any
import solve_q1 as model


def solve_scenario(data):
    t0=time.perf_counter(); result=model.solve_lp(data); checks=model.calculate_checks(result,data)
    if checks['mutual_exclusion_violation']>model.TOL: result=model.solve_milp(data); checks=model.calculate_checks(result,data)
    return result,checks,time.perf_counter()-t0


def metrics(result,checks,data,runtime):
    prices=data['price']; loads=data['load_energy']; pv=data['pv_energy']; q,c,s,w,E=result['q'],result['c'],result['s'],result['w'],result['E']
    benchmark=sum(prices[i]*max(loads[i]-pv[i],0) for i in range(model.T)); objective=checks['objective']; total_q=checks['total_purchase']
    return {
      'objective':objective,'benchmark_cost':benchmark,'saving_cost':benchmark-objective,'saving_percent':100*(benchmark-objective)/benchmark,
      'average_cost_per_purchased_kwh':objective/total_q,'average_cost_per_load_kwh':objective/sum(loads),'total_load':sum(loads),'total_pv':sum(pv),
      'total_purchase':total_q,'total_charge':checks['total_charge'],'total_discharge':checks['total_discharge'],
      'charge_discharge_loss':checks['total_charge']-checks['total_discharge'],'total_curtailment':checks['total_curtailment'],
      'pv_utilization_percent':100*(sum(pv)-checks['total_curtailment'])/sum(pv),'soc_min':float(min(E)),'soc_max':float(max(E)),
      'soc_initial':float(E[0]),'soc_terminal':float(E[-1]),'max_charge_power':checks['max_charge_power'],'max_discharge_power':checks['max_discharge_power'],
      'charge_power_utilization_percent':100*checks['max_charge_power']/model.POWER_MAX,'discharge_power_utilization_percent':100*checks['max_discharge_power']/model.POWER_MAX,
      'balance_residual_max':checks['balance_residual_max'],'state_residual_max':checks['state_residual_max'],'mutual_exclusion_violation':checks['mutual_exclusion_violation'],
      'zero_purchase_intervals':int(sum(abs(v)<=model.TOL for v in q)),'zero_charge_intervals':int(sum(abs(v)<=model.TOL for v in c)),'zero_discharge_intervals':int(sum(abs(v)<=model.TOL for v in s)),
      'soc_min_binding_count':int(sum(abs(v-model.SOC_MIN)<=model.TOL for v in E)),'soc_max_binding_count':int(sum(abs(v-model.SOC_MAX)<=model.TOL for v in E)),
      'runtime_seconds':runtime,'specified_purchase':{k:float(q[i-1]) for k,i in {'10:00-10:10':61,'12:00-12:10':73,'14:00-14:10':85,'16:00-16:10':97,'18:00-18:10':109,'20:00-20:10':121}.items()},
      'four_hour_blocks':{f'block_{b+1}':{'charge':float(sum(c[b*24:(b+1)*24])),'discharge':float(sum(s[b*24:(b+1)*24]))} for b in range(6)}
    }


def eta_sensitivity(data):
    rows=[]; old=model.ETA
    try:
        for label,eta in [('single_0.80',.80),('single_0.85',.85),('single_0.90',.90),('single_0.95',.95),('single_1.00',1.0),('roundtrip_0.90',math.sqrt(.9))]:
            model.ETA=eta; result,checks,_=solve_scenario(data)
            rows.append({'scenario':label,'eta_charge':eta,'eta_discharge':eta,'roundtrip_efficiency':eta*eta,'objective':checks['objective'],'total_purchase':checks['total_purchase'],'total_charge':checks['total_charge'],'total_discharge':checks['total_discharge'],'method':result['method'],'feasible':all(checks[k]<=model.TOL for k in ['balance_residual_max','state_residual_max','soc_bound_violation','charge_power_violation','discharge_power_violation','nonnegative_violation','curtailment_violation','mutual_exclusion_violation','terminal_soc_error'])})
    finally: model.ETA=old
    return rows


def soc_sensitivity(data):
    rows=[]; old=model.SOC_INITIAL
    try:
        for soc in (5400.,6000.,6600.):
            model.SOC_INITIAL=soc; result,checks,_=solve_scenario(data)
            rows.append({'initial_and_terminal_soc':soc,'objective':checks['objective'],'total_purchase':checks['total_purchase'],'total_charge':checks['total_charge'],'total_discharge':checks['total_discharge'],'method':result['method'],'feasible':checks['terminal_soc_error']<=model.TOL})
    finally: model.SOC_INITIAL=old
    return rows

def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def dual_rows(result):
    d=result['diagnostics']; rows=[]
    for t in range(model.T):
        start,end=model.canonical_interval(t+1)
        rows.append({'t':t+1,'interval':f'{start}-{end}','balance_marginal':d['balance_marginal'][t],'state_marginal':d['state_marginal'][t],
          'soc_start_lower_marginal':d['soc_lower_marginal'][t],'soc_start_upper_marginal':d['soc_upper_marginal'][t],
          'soc_end_lower_marginal':d['soc_lower_marginal'][t+1],'soc_end_upper_marginal':d['soc_upper_marginal'][t+1],
          'charge_upper_marginal':d['charge_upper_marginal'][t],'discharge_upper_marginal':d['discharge_upper_marginal'][t]})
    return rows


def report(metrics,eta,soc,attachment):
    lines=['# 问题一模型评估与敏感性分析','',f'- 输入附件：`{attachment}`',f'- 输入 SHA-256：`{model.sha256_file(attachment)}`','','## 核心指标','',
      f'- 最优购电费：{metrics["objective"]:.9f} 元。',f'- 无储能基准费用：{metrics["benchmark_cost"]:.9f} 元。',f'- 节省比例：{metrics["saving_percent"]:.6f}%。',
      f'- 最大功率平衡残差：{metrics["balance_residual_max"]:.6e} kWh。',f'- 最大储能递推残差：{metrics["state_residual_max"]:.6e} kWh。','',
      '| 指标 | 数值 |','|---|---:|',f'| 全天购电量/kWh | {metrics["total_purchase"]:.9f} |',f'| 充电量/kWh | {metrics["total_charge"]:.9f} |',
      f'| 放电量/kWh | {metrics["total_discharge"]:.9f} |',f'| 充放电损耗/kWh | {metrics["charge_discharge_loss"]:.9f} |',f'| 弃光量/kWh | {metrics["total_curtailment"]:.9f} |',
      f'| 最低/最高储电量/kWh | {metrics["soc_min"]:.3f}/{metrics["soc_max"]:.3f} |','','## 效率敏感性','','| 情景 | 单程效率 | 往返效率 | 最优费用/元 | 购电量/kWh | 可行 |','|---|---:|---:|---:|---:|---|']
    for r in eta: lines.append(f'| {r["scenario"]} | {r["eta_charge"]:.6f} | {r["roundtrip_efficiency"]:.6f} | {r["objective"]:.6f} | {r["total_purchase"]:.6f} | {r["feasible"]} |')
    lines += ['','## 初始和结束储电量敏感性','','| 储电量/kWh | 最优费用/元 | 购电量/kWh | 可行 |','|---:|---:|---:|---|']
    for r in soc: lines.append(f'| {r["initial_and_terminal_soc"]:.3f} | {r["objective"]:.6f} | {r["total_purchase"]:.6f} | {r["feasible"]} |')
    lines += ['','## 对偶信息','','`q1_duals.csv` 保存 SciPy 原始边际值；论文引用前需结合对偶符号约定解释。','']
    return '\n'.join(lines)


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--attachment',type=Path,default=model.DEFAULT_ATTACHMENT); p.add_argument('--output-dir',type=Path,default=model.Q1_ROOT); args=p.parse_args(argv)
    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True); data=model.load_attachment(args.attachment)
    result,checks,runtime=solve_scenario(data); m=metrics(result,checks,data,runtime); eta=eta_sensitivity(data); soc=soc_sensitivity(data); duals=dual_rows(result)
    payload={'status':'verified','attachment':str(args.attachment.resolve()),'attachment_sha256':model.sha256_file(args.attachment),'metrics':m,'eta_sensitivity':eta,'soc_sensitivity':soc}
    (out/'q1_evaluation.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True),encoding='utf-8')
    write_csv(out/'q1_sensitivity_eta.csv',eta); write_csv(out/'q1_sensitivity_soc.csv',soc); write_csv(out/'q1_duals.csv',duals)
    (out/'q1_evaluation.md').write_text(report(m,eta,soc,args.attachment),encoding='utf-8')
    print(f'objective={m["objective"]:.12f}'); print(f'saving_percent={m["saving_percent"]:.9f}'); print(out/'q1_evaluation.md'); return 0

if __name__=='__main__': raise SystemExit(main())
