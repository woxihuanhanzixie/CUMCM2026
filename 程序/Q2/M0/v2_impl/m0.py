#!/usr/bin/env python3
"""Q2 v2 M0_V0C (spec 02): 0.8-quantile risk-adjusted midnight LP + frozen greedy.

Frozen definition:
  n_hat = k0 ridge net forecast (kWh/slot), published at midnight of day d.
  delta_dt = empirical 0.8 quantile (linear interpolation) of historical k0
  forecast errors e_hj = net_act[h,j] - n_hat(h,j|h), pooled over publish days
  h in [Jan22, d) and slots j in [t-2, t+2].
  a_dt = n_hat_dt + delta_dt.  Midnight LP (alpha=.8, R=1200 = E_LO):
    min sum p_t q_t
    s.t. q - c + s - w = a ;  E_t = E_{t-1} + .9c - s/.9 ;  1200 <= E <= 10800
         q >= 0 ; 0 <= c,s <= 5000/6 ; 0 <= w <= PVhat/6 (= PVhat kWh/slot)
  No day-end = day-start constraint, no terminal value (Phi = 0).
  Only q is locked; plan c/s/w/E are archived, never executed.
  LP infeasible / no verified incumbent -> shared fallback q = max(n_hat, 0),
  full input recorded (spec 01 §5). Actual day: frozen greedy, plan fee
  sum p*q, emergency fee 5*sum p*z.

Usage:
  python Q2/v2_impl/m0.py --solve            # annual run -> outputs/M0_V0C
  python Q2/v2_impl/m0.py --check            # independent audit of outputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import OptimizeWarning, linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v2_impl.common import result_writer  # noqa: E402
from v2_impl.common.data_io import (DT, ETA_C, ETA_D, E_HI, E_LO, E_PORT_MAX,  # noqa: E402
                                    EMERGENCY_MULT, N_DAYS, Q2_ROOT, SCORED_START,
                                    T, load_attachment1, load_attachment2)
from v2_impl.common.environment import greedy_step, simulate_day  # noqa: E402
from v2_impl.common.forecast import RidgeForecaster  # noqa: E402
from v2_impl.common.january import check_january, run_january  # noqa: E402
from v2_impl.common.ledger import audit  # noqa: E402
from v2_impl.common.result_writer import check_result2, write_result2  # noqa: E402

ALPHA = 0.8
HALF_WIDTH = 2
POOL_START = 21            # Jan22, day index (Jan1 = 0)
SOLVE_TOL = 1e-7
TIME_LIMIT = 10.0
FEAS_TOL = 1e-6
SEED = 20260911
MODEL = 'M0_V0C'
OUT_DIR = Q2_ROOT / 'outputs' / MODEL
PERTURB_DATES = ((31, 'Feb1'), (78, 'Mar20'), (171, 'Jun21'),
                 (265, 'Sep23'), (354, 'Dec21'))
PERTURB_A, PERTURB_B = 1.37, 17.0


# ---------------------------------------------------------------- plan LP

def build_plan_lp(a, p, wcap, e0):
    """n-slot plan LP. Variables: q, c, s, w, E (5n); 2n equality rows."""
    n = len(a)
    off = {'q': 0, 'c': n, 's': 2 * n, 'w': 3 * n, 'E': 4 * n}
    a = np.asarray(a, float)
    p = np.asarray(p, float)
    wcap = np.asarray(wcap, float)
    Aeq = np.zeros((2 * n, 5 * n))
    beq = np.zeros(2 * n)
    for t in range(n):
        Aeq[t, t] = 1.0
        Aeq[t, off['c'] + t] = -1.0
        Aeq[t, off['s'] + t] = 1.0
        Aeq[t, off['w'] + t] = -1.0
        beq[t] = a[t]
        Aeq[n + t, off['E'] + t] = 1.0
        Aeq[n + t, off['c'] + t] = -ETA_C
        Aeq[n + t, off['s'] + t] = 1.0 / ETA_D
        if t > 0:
            Aeq[n + t, off['E'] + t - 1] = -1.0
        else:
            beq[n + t] = float(e0)
    obj = np.zeros(5 * n)
    obj[off['q']:off['q'] + n] = p
    bounds = [(0.0, None)] * n + [(0.0, E_PORT_MAX)] * n + [(0.0, E_PORT_MAX)] * n \
        + [(0.0, float(wcap[t])) for t in range(n)] + [(E_LO, E_HI)] * n
    return obj, Aeq, beq, bounds, off


def check_plan_solution(a, p, wcap, e0, x, off, tol=FEAS_TOL):
    """Independent feasibility check of a raw LP solution (spec 01 incumbent rule)."""
    n = len(a)
    q = x[off['q']:off['q'] + n]
    c = x[off['c']:off['c'] + n]
    s = x[off['s']:off['s'] + n]
    w = x[off['w']:off['w'] + n]
    E = x[off['E']:off['E'] + n]
    bal = q - c + s - w - np.asarray(a)
    eres = np.empty(n)
    eres[0] = E[0] - float(e0) - ETA_C * c[0] + s[0] / ETA_D
    eres[1:] = E[1:] - E[:-1] - ETA_C * c[1:] + s[1:] / ETA_D
    viol: list[str] = []
    worst = max(float(np.max(np.abs(bal))), float(np.max(np.abs(eres))))
    if worst > tol:
        viol.append(f'max_eq_residual {worst:.3g}')
    if np.min(q) < -tol:
        viol.append('q negative')
    if np.min(c) < -tol or np.max(c) > E_PORT_MAX + tol:
        viol.append('c port bound')
    if np.min(s) < -tol or np.max(s) > E_PORT_MAX + tol:
        viol.append('s port bound')
    if np.min(w) < -tol or np.any(w > np.asarray(wcap) + tol):
        viol.append('w bound')
    if np.min(E) < E_LO - tol or np.max(E) > E_HI + tol:
        viol.append('E bound')
    return viol, worst, (q, c, s, w, E)


def solve_plan_lp(a, p, wcap, e0, time_limit=TIME_LIMIT):
    """Solve the plan LP; return usable solution only if independently feasible."""
    obj, Aeq, beq, bounds, off = build_plan_lp(a, p, wcap, e0)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', OptimizeWarning)
        res = linprog(obj, A_eq=Aeq, b_eq=beq, bounds=bounds, method='highs',
                      options={'primal_feasibility_tolerance': SOLVE_TOL,
                               'dual_feasibility_tolerance': SOLVE_TOL,
                               'time_limit': time_limit,
                               'random_seed': SEED})
    wall = time.perf_counter() - t0
    out = {'status': int(res.status), 'status_text': str(res.message).strip(),
           'wall_s': wall, 'nit': getattr(res, 'nit', None), 'usable': False,
           'max_residual': None}
    if res.status in (0, 1) and res.x is not None:
        viol, worst, parts = check_plan_solution(a, p, wcap, e0, res.x, off)
        out['max_residual'] = worst
        if not viol:
            out['usable'] = True
            out['q'], out['c'], out['s'], out['w'], out['E'] = parts
            out['fun'] = float(np.dot(obj, res.x))
        else:
            out['violations'] = viol
    return out


def fallback_q(n_hat):
    return np.maximum(np.asarray(n_hat, float), 0.0)


# ---------------------------------------------------------------- M0 model

def quantile_delta(err, alpha=ALPHA):
    """delta_t = alpha-quantile over h and j in [t-2, t+2]; err: (nh, T)."""
    err = np.asarray(err, float)
    nh, Tn = err.shape
    out = np.empty(Tn)
    for t in range(1, Tn + 1):
        lo = max(0, t - 3)
        hi = min(Tn, t + 2)
        out[t - 1] = float(np.quantile(err[:, lo:hi], alpha))
    return out


class M0Planner:
    def __init__(self, data, fc):
        self.price = np.asarray(data['price'], float)
        self.net_act = (np.asarray(data['load_act'], float)
                        - np.asarray(data['pv_act'], float)) * DT
        self.dates = data['dates']
        self.fc = fc
        self._nhat: dict[int, np.ndarray] = {}
        self._pvhat: dict[int, np.ndarray] = {}

    def nhat_at(self, h: int) -> np.ndarray:
        """k0 ridge forecast at publish day h (cached; archive shared by models)."""
        if h not in self._nhat:
            pr = self.fc.predict_day(h, 0)
            self._pvhat[h] = np.clip(pr['pv'], 0.0, None) * DT
            self._nhat[h] = (pr['load'] - pr['pv']) * DT
        return self._nhat[h]

    def forecast_archive(self, days) -> None:
        for h in days:
            self.nhat_at(h)

    def delta_dt(self, d: int):
        """delta for day d from errors of publish days [POOL_START, d)."""
        nh = d - POOL_START
        err = np.empty((nh, T))
        for i, h in enumerate(range(POOL_START, d)):
            err[i] = self.net_act[h] - self.nhat_at(h)
        return quantile_delta(err), nh

    def plan_midnight(self, d: int, e0: float,
                      a_override=None, wcap_override=None) -> dict:
        """Midnight decision for day d: forecast, delta, a, LP -> locked q + evidence."""
        nhat = self.nhat_at(d)
        wcap = self._pvhat[d] if wcap_override is None else np.asarray(wcap_override, float)
        delta, pool_size = self.delta_dt(d)
        a = nhat + delta if a_override is None else np.asarray(a_override, float)
        warn = np.where(a < -wcap)[0]
        lp = solve_plan_lp(a, self.price, wcap, e0)
        rec = {'day': d, 'n_hat': nhat, 'pv_hat': self._pvhat[d], 'delta': delta,
               'a': a, 'wcap': wcap, 'pool_size': pool_size,
               'warn_slots': warn.tolist(), 'lp_status': lp['status'],
               'lp_status_text': lp['status_text'], 'wall_s': lp['wall_s'],
               'nit': lp['nit'], 'max_residual': lp['max_residual'],
               'fallback': not lp['usable']}
        if lp['usable']:
            rec['q'] = lp['q']
            rec['plan_c'], rec['plan_s'] = lp['c'], lp['s']
            rec['plan_w'], rec['plan_E'] = lp['w'], lp['E']
            rec['lp_fun'] = lp['fun']
            rec['simul_cs_plan'] = int(np.sum((lp['c'] > 1e-6) & (lp['s'] > 1e-6)))
        else:
            rec['q'] = fallback_q(nhat)
            rec['plan_c'] = rec['plan_s'] = rec['plan_w'] = rec['plan_E'] = None
            rec['lp_fun'] = None
            rec['simul_cs_plan'] = 0
            rec['fallback_reason'] = (f'status {lp["status"]} '
                                      f'({lp["status_text"]})' +
                                      ('; ' + '; '.join(lp.get('violations', []))
                                       if lp.get('violations') else ''))
        return rec

    def run_day(self, d: int, e0: float, **overrides) -> dict:
        rec = self.plan_midnight(d, e0, **overrides)
        sim = simulate_day(self.net_act[d], rec['q'], e0, price=self.price)
        rec.update({'c': sim['c'], 's': sim['s'], 'z': sim['z'], 'w': sim['w'],
                    'E': sim['E'], 'e_end': sim['e_end'], 'e0': float(e0),
                    'emg_cost': sim['emg_cost']})
        rec['plan_cost'] = float(np.dot(self.price, rec['q']))
        rec['total_cost'] = rec['plan_cost'] + rec['emg_cost']
        return rec


# ---------------------------------------------------------------- annual run

def _month_key(dt):
    return f'{dt.year:04d}-{dt.month:02d}'


def run_annual(out_dir: Path = OUT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    a1 = load_attachment1()
    a2 = load_attachment2()
    price = a1['price']
    data = {'price': price, 'load_act': a2['load_act'], 'pv_act': a2['pv_act'],
            'dates': a2['dates']}
    jan = run_january(price, a2['load_act'], a2['pv_act'])
    jan_errs = check_january(jan)
    fc = RidgeForecaster(data)
    planner = M0Planner(data, fc)
    planner.forecast_archive(range(POOL_START, N_DAYS))

    e = float(jan['feb1_energy'])
    recs: list[dict] = []
    audit_errs_total: list[str] = []
    for d in range(SCORED_START, N_DAYS):
        rec = planner.run_day(d, e)
        claimed = {'z': rec['z'], 'w': rec['w'], 'E': rec['E'],
                   'e_end': rec['e_end'], 'emg_cost': rec['emg_cost']}
        errs, _r = audit(planner.net_act[d], rec['q'], rec['c'], rec['s'], e,
                         claimed, price)
        for msg in errs:
            audit_errs_total.append(f'day {d}: {msg}')
        rec['audit_errors'] = errs
        recs.append(rec)
        e = rec['e_end']

    days = np.arange(SCORED_START, N_DAYS)
    q = np.stack([r['q'] for r in recs])
    c = np.stack([r['c'] for r in recs])
    s = np.stack([r['s'] for r in recs])
    z = np.stack([r['z'] for r in recs])
    w = np.stack([r['w'] for r in recs])
    E = np.stack([r['E'] for r in recs])
    plan_cost = np.array([r['plan_cost'] for r in recs])
    emg_cost = np.array([r['emg_cost'] for r in recs])
    tot_cost = plan_cost + emg_cost
    dates = data['dates'][SCORED_START:N_DAYS]

    # ---- write result2_candidate.xlsx + readback
    wb_path = out_dir / 'result2_candidate.xlsx'
    write_result2(wb_path, dates, q, c, s, z, E, price)
    wb_errs = check_result2(wb_path, dates, q, c, s, z, E, price)

    # ---- trajectory / summaries / orders / refs / logs
    _write_trajectory(out_dir, days, dates, q, c, s, z, w, E, price)
    _write_summaries(out_dir, days, dates, plan_cost, emg_cost, tot_cost, E, w,
                     recs, jan['jan_cost'])
    _write_orders_and_refs(out_dir, days, dates, recs, planner)
    _write_solver_and_fallback_logs(out_dir, days, dates, recs, price)

    # ---- perturbation checks (midnight view; M0 has no intraday re-optimization)
    pert = perturbation_check(planner, recs)

    # ---- independent audit (ledger is the independent implementation)
    resid_cons = float(np.abs(q - c + s + z - w
                              - planner.net_act[SCORED_START:N_DAYS]).max())
    eres = np.empty_like(E)
    eres[:, 0] = E[:, 0] - np.array([r['e0'] for r in recs]) - ETA_C * c[:, 0] + s[:, 0] / ETA_D
    eres[:, 1:] = E[:, 1:] - E[:, :-1] - ETA_C * c[:, 1:] + s[:, 1:] / ETA_D
    resid_E = float(np.abs(eres).max())
    sim_cs = int(np.sum((c > 1e-6) & (s > 1e-6)))
    sim_cz = int(np.sum((c > 1e-6) & (z > 1e-6)))
    e0_cont_ok = all(abs(recs[i + 1]['e0'] - recs[i]['e_end']) < 1e-9
                     for i in range(len(recs) - 1))
    plan_re = float(np.dot(price, q.sum(axis=0)))
    emg_re = EMERGENCY_MULT * float(np.dot(price, z.sum(axis=0)))
    cost_diff = max(abs(plan_re - plan_cost.sum()), abs(emg_re - emg_cost.sum()))
    n_fallback = int(sum(r['fallback'] for r in recs))
    wall = np.array([r['wall_s'] for r in recs])

    audit_report = {
        'model': MODEL,
        'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'candidate_solved': True,
        'independent_audit_passed': False,
        'user_adopted': False,
        'optimizer_fully_met': n_fallback == 0,
        'hard_acceptance': {
            'conservation_residual_max_kwh': resid_cons,
            'E_recursion_residual_max_kwh': resid_E,
            'residual_le_1e-6': bool(resid_cons <= 1e-6 and resid_E <= 1e-6),
            'cost_recompute_diff_yuan': float(cost_diff),
            'cost_diff_le_0.01': bool(cost_diff <= 0.01),
            'bounds_and_flows_ok': not audit_errs_total and sim_cs == 0 and sim_cz == 0,
        },
        'counts': {'days': int(len(recs)), 'fallbacks': n_fallback,
                   'simultaneous_c_s_actual': sim_cs,
                   'simultaneous_c_z_actual': sim_cz,
                   'plan_layer_simultaneous_c_s_slots':
                       int(sum(r['simul_cs_plan'] for r in recs)),
                   'audit_ledger_errors': len(audit_errs_total)},
        'e0_continuity_ok': e0_cont_ok,
        'january_check': {'errors': jan_errs, 'jan_cost': jan['jan_cost'],
                          'feb1_energy': jan['feb1_energy']},
        'perturbation': pert,
        'workbook_readback': {'errors': wb_errs},
        'solve_time_s': {'avg': float(wall.mean()), 'p95': float(np.percentile(wall, 95)),
                         'max': float(wall.max())},
        'total_wall_s': time.perf_counter() - t_start,
        'environment': _env_record(),
        'ledger_errors': audit_errs_total[:50],
        'note_m0_intraday': ('M0 has no intraday re-optimization; slot actions follow '
                             'the frozen greedy formula from locked q and revealed '
                             'actuals only (verified by construction, see '
                             'perturbation.intraday_slot73).'),
    }
    audit_report['independent_audit_passed'] = bool(
        audit_report['hard_acceptance']['residual_le_1e-6']
        and audit_report['hard_acceptance']['cost_diff_le_0.01']
        and audit_report['hard_acceptance']['bounds_and_flows_ok']
        and e0_cont_ok and not wb_errs
        and pert['midnight_5_dates_ok'] and pert['intraday_slot73_ok'])
    with (out_dir / 'independent_audit.json').open('w', encoding='utf-8') as f:
        json.dump(audit_report, f, ensure_ascii=False, indent=2)

    _write_reproduce(out_dir, audit_report)
    return {'jan': jan['jan_cost'], 'plan': float(plan_cost.sum()),
            'emg': float(emg_cost.sum()), 'total': float(tot_cost.sum()),
            'jan_plus_total': jan['jan_cost'] + float(tot_cost.sum()),
            'fallbacks': n_fallback, 'audit': audit_report,
            'wall_s': audit_report['total_wall_s'],
            'e_end': float(recs[-1]['e_end'])}


def perturbation_check(planner: M0Planner, recs: list[dict]) -> dict:
    """Five fixed dates: perturb L/G for day d and later (x1.37+17); the midnight
    plan uses only days < d, so forecast, delta, a, LP and locked q must be
    bit-identical. Intraday slot 73 action uses only revealed inputs."""
    results = []
    for d, name in PERTURB_DATES:
        idx = d - SCORED_START
        load_p = planner.fc.load_act.copy()
        pv_p = planner.fc.pv_act.copy()
        load_p[d:] = load_p[d:] * PERTURB_A + PERTURB_B
        pv_p[d:] = pv_p[d:] * PERTURB_A + PERTURB_B
        data_p = {'price': planner.price,
                  'load_act': load_p, 'pv_act': pv_p, 'dates': planner.dates}
        fc_p = RidgeForecaster(data_p)
        pl_p = M0Planner(data_p, fc_p)
        rec_p = pl_p.plan_midnight(d, recs[idx]['e0'])
        base = recs[idx]
        q_same = bool(np.array_equal(rec_p['q'], base['q']))
        a_same = bool(np.array_equal(rec_p['a'], base['a']))
        results.append({'date': name, 'day_index': d, 'q_identical': q_same,
                        'a_identical': a_same})
        if not (q_same and a_same):
            results[-1]['note'] = 'midnight perturbation changed the plan'
    slot73_ok = True
    for d, name in PERTURB_DATES:
        idx = d - SCORED_START
        r = recs[idx]
        e = r['e0']
        c_out = np.empty(73)
        z_out = np.empty(73)
        for t in range(73):
            st = greedy_step(e, planner.net_act[d][t], r['q'][t])
            c_out[t] = st['c']
            z_out[t] = st['z']
            e = st['e']
        same = (np.array_equal(c_out, r['c'][:73])
                and np.array_equal(z_out, r['z'][:73]))
        slot73_ok = slot73_ok and bool(same)
    return {'midnight_5_dates_ok': all(r['q_identical'] for r in results),
            'intraday_slot73_ok': slot73_ok, 'dates': results}


def _env_record() -> dict:
    return {'python': sys.version.split()[0], 'numpy': np.__version__,
            'scipy': scipy.__version__, 'solver': 'scipy linprog method=highs',
            'seed': SEED, 'time_limit_s': TIME_LIMIT,
            'lp_tolerances': {'primal': SOLVE_TOL, 'dual': SOLVE_TOL},
            'machine': platform.machine(), 'processor': platform.processor(),
            'reference_env_note': 'reference environment SciPy 1.15.3 / NumPy 1.26.4'}


def _write_trajectory(out_dir, days, dates, q, c, s, z, w, E, price) -> None:
    with (out_dir / 'trajectory.csv').open('w', encoding='utf-8', newline='') as f:
        f.write('day_index,date,slot,q_kwh,c_kwh,s_kwh,z_kwh,w_kwh,E_kwh,price\n')
        for i, d in enumerate(days):
            for t in range(T):
                f.write(f'{d},{dates[i].isoformat()},{t + 1},{q[i, t]:.12g},'
                        f'{c[i, t]:.12g},{s[i, t]:.12g},{z[i, t]:.12g},'
                        f'{w[i, t]:.12g},{E[i, t]:.12g},{price[t]:.12g}\n')


def _write_summaries(out_dir, days, dates, plan_cost, emg_cost, tot_cost, E, w,
                     recs, jan_cost) -> None:
    with (out_dir / 'daily_summary.csv').open('w', encoding='utf-8', newline='') as f:
        f.write('day_index,date,plan_cost,emg_cost,total_cost,e0,e_end,'
                'lp_status,lp_usable,wall_s,fallback,n_warn_slots,'
                'simul_cs_plan,pool_size\n')
        for i, r in enumerate(recs):
            f.write(f'{r["day"]},{dates[i].isoformat()},{plan_cost[i]:.12g},'
                    f'{emg_cost[i]:.12g},{tot_cost[i]:.12g},{r["e0"]:.12g},'
                    f'{r["e_end"]:.12g},{r["lp_status"]},'
                    f'{int(not r["fallback"])},'
                    f'{r["wall_s"]:.6g},{int(r["fallback"])},{len(r["warn_slots"])},'
                    f'{r["simul_cs_plan"]},{r["pool_size"]}\n')
    months: dict[str, dict] = {}
    for i, r in enumerate(recs):
        m = _month_key(dates[i])
        b = months.setdefault(m, {'plan': 0.0, 'emg': 0.0, 'E': [], 'w': 0.0})
        b['plan'] += plan_cost[i]
        b['emg'] += emg_cost[i]
        b['E'].append(E[i])
        b['w'] += float(w[i].sum())
    with (out_dir / 'monthly_summary.csv').open('w', encoding='utf-8', newline='') as f:
        f.write('month,plan_cost,emg_cost,total_cost,e_end,min_E,max_E,unused_w\n')
        for m, b in months.items():
            arr = np.vstack(b['E'])
            f.write(f'{m},{b["plan"]:.12g},{b["emg"]:.12g},'
                    f'{b["plan"] + b["emg"]:.12g},{arr[-1, -1]:.12g},'
                    f'{arr.min():.12g},{arr.max():.12g},{b["w"]:.12g}\n')
    with (out_dir / 'annual_summary.csv').open('w', encoding='utf-8', newline='') as f:
        f.write('metric,value\n')
        f.write(f'plan_cost_yuan,{plan_cost.sum():.12g}\n')
        f.write(f'emg_cost_yuan,{emg_cost.sum():.12g}\n')
        f.write(f'total_cost_yuan,{tot_cost.sum():.12g}\n')
        f.write(f'jan_cost_yuan,{jan_cost:.12g}\n')
        f.write(f'jan_plus_total_yuan,{jan_cost + tot_cost.sum():.12g}\n')
        f.write(f'unused_energy_kwh,{float(w.sum()):.12g}\n')
        f.write(f'total_charge_kwh,{float(np.array([r["c"] for r in recs]).sum()):.12g}\n')
        f.write(f'total_discharge_kwh,{float(np.array([r["s"] for r in recs]).sum()):.12g}\n')
        f.write(f'total_emergency_kwh,{float(np.array([r["z"] for r in recs]).sum()):.12g}\n')
        f.write(f'min_E_kwh,{E.min():.12g}\n')
        f.write(f'max_E_kwh,{E.max():.12g}\n')
        f.write(f'year_end_E_kwh,{float(recs[-1]["e_end"]):.12g}\n')


def _write_orders_and_refs(out_dir, days, dates, recs, planner) -> None:
    with (out_dir / 'midnight_orders.csv').open('w', encoding='utf-8', newline='') as f:
        f.write('day_index,date,' + ','.join(str(t) for t in range(1, T + 1)) + '\n')
        for i, r in enumerate(recs):
            f.write(f'{r["day"]},{dates[i].isoformat()},'
                    + ','.join(f'{v:.12g}' for v in r['q']) + '\n')
    with (out_dir / 'delta_pool.csv').open('w', encoding='utf-8', newline='') as f:
        f.write('day_index,date,' + ','.join(str(t) for t in range(1, T + 1)) + '\n')
        for i, r in enumerate(recs):
            f.write(f'{r["day"]},{dates[i].isoformat()},'
                    + ','.join(f'{v:.12g}' for v in r['delta']) + '\n')
    with (out_dir / 'forecast_refs.csv').open('w', encoding='utf-8', newline='') as f:
        f.write('day_index,date,k,n_train,source,pool_size,delta_min,delta_mean,'
                'delta_max,n_warn_slots,plan_cost_locked\n')
        for i, r in enumerate(recs):
            ntr = planner.fc.n_train(r['day'], 0)
            f.write(f'{r["day"]},{dates[i].isoformat()},0,{ntr},'
                    f'{"ridge" if ntr >= 7 else "fallback_S"},{r["pool_size"]},'
                    f'{float(r["delta"].min()):.12g},{float(r["delta"].mean()):.12g},'
                    f'{float(r["delta"].max()):.12g},{len(r["warn_slots"])},'
                    f'{r["plan_cost"]:.12g}\n')
    with (out_dir / 'plan_aux.csv').open('w', encoding='utf-8', newline='') as f:
        f.write('day_index,slot,c_plan,s_plan,w_plan,E_plan\n')
        for i, r in enumerate(recs):
            if r['plan_c'] is None:
                continue
            for t in range(T):
                f.write(f'{r["day"]},{t + 1},{r["plan_c"][t]:.12g},{r["plan_s"][t]:.12g},'
                        f'{r["plan_w"][t]:.12g},{r["plan_E"][t]:.12g}\n')


def _write_solver_and_fallback_logs(out_dir, days, dates, recs, price) -> None:
    with (out_dir / 'solver_log.csv').open('w', encoding='utf-8', newline='') as f:
        f.write('day_index,date,status,status_text,usable,wall_s,objective_lp,'
                'locked_plan_cost,max_eq_residual,simul_cs_plan,pool_size,'
                'n_warn_slots,input_sha256,nit\n')
        for i, r in enumerate(recs):
            digest = hashlib.sha256()
            for arr in (r['a'], price, r['wcap']):
                digest.update(np.asarray(arr, float).tobytes())
            digest.update(np.array([r['e0']], float).tobytes())
            obj = f'{r["lp_fun"]:.12g}' if r['lp_fun'] is not None else ''
            res = f'{r["max_residual"]:.3g}' if r['max_residual'] is not None else ''
            nit = r['nit'] if r['nit'] is not None else ''
            f.write(f'{r["day"]},{dates[i].isoformat()},{r["lp_status"]},'
                    f'"{r["lp_status_text"]}",{int(not r["fallback"])},{r["wall_s"]:.6g},'
                    f'{obj},{r["plan_cost"]:.12g},{res},{r["simul_cs_plan"]},'
                    f'{r["pool_size"]},{len(r["warn_slots"])},{digest.hexdigest()[:16]},'
                    f'{nit}\n')
    fb_dir = out_dir / 'fallback_inputs'
    with (out_dir / 'fallback_log.csv').open('w', encoding='utf-8', newline='') as f:
        f.write('day_index,date,status,reason,input_npz\n')
        for i, r in enumerate(recs):
            if not r['fallback']:
                continue
            fb_dir.mkdir(exist_ok=True)
            path = fb_dir / f'input_day{r["day"]:03d}.npz'
            np.savez(path, a=r['a'], p=price, wcap=r['wcap'],
                     e0=np.array([r['e0']]), n_hat=r['n_hat'], delta=r['delta'])
            f.write(f'{r["day"]},{dates[i].isoformat()},{r["lp_status"]},'
                    f'"{r.get("fallback_reason", "")}",{path.name}\n')


def _write_reproduce(out_dir, audit_report) -> None:
    import v2_impl
    import v2_impl.common.data_io as dio
    lines = [
        '# M0_V0C reproduce',
        '',
        '## commands',
        '```',
        'python Q2/v2_impl/m0.py --solve',
        'python Q2/v2_impl/m0.py --check',
        'python -m unittest discover -s Q2/v2_impl/tests -v',
        '```',
        '',
        '## frozen inputs (read-only, SHA-256)',
        f'attachment1: {dio.sha256_file(dio.A1_DEFAULT)}',
        f'attachment2: {dio.sha256_file(dio.A2_DEFAULT)}',
        '',
        '## code (SHA-256)',
        f'm0.py: {dio.sha256_file(Path(__file__).resolve())}',
        f'common/data_io.py: {dio.sha256_file(Path(dio.__file__).resolve())}',
        f'common/forecast.py: {dio.sha256_file(Path(v2_impl.__file__).resolve().parent / "common" / "forecast.py")}',
        f'common/environment.py: {dio.sha256_file(Path(v2_impl.__file__).resolve().parent / "common" / "environment.py")}',
        f'common/ledger.py: {dio.sha256_file(Path(v2_impl.__file__).resolve().parent / "common" / "ledger.py")}',
        f'common/result_writer.py: {dio.sha256_file(Path(v2_impl.__file__).resolve().parent / "common" / "result_writer.py")}',
        f'common/january.py: {dio.sha256_file(Path(v2_impl.__file__).resolve().parent / "common" / "january.py")}',
        '',
        '## environment',
        f'python {audit_report["environment"]["python"]}, numpy '
        f'{audit_report["environment"]["numpy"]}, scipy {audit_report["environment"]["scipy"]}',
        f'HiGHS via scipy linprog (method=highs), LP primal/dual tol 1e-7, '
        f'time limit {TIME_LIMIT}s, seed {SEED}',
        '',
        '## frozen model parameters (spec 02 / V2 §5)',
        'alpha=0.8, R=1200 (=E_LO), residual neighborhood ±2 slots,',
        'error pool from Jan22 (day index 21), linear-interpolation empirical quantile,',
        'plan w cap = PV forecast energy per slot (PVhat/6), no day-end constraint,',
        'January initialization per spec 01 (Feb1 E0 = 10800), no parameter search.',
        '',
        '## status fields (spec 05: three separate fields)',
        f'candidate_solved = {audit_report["candidate_solved"]}',
        f'independent_audit_passed = {audit_report["independent_audit_passed"]}',
        f'user_adopted = {audit_report["user_adopted"]}',
        f'optimizer_fully_met = {audit_report["optimizer_fully_met"]} '
        f'(fallbacks = {audit_report["counts"]["fallbacks"]})',
    ]
    (out_dir / 'reproduce.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


# ---------------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser(description='Q2 v2 M0_V0C solver')
    ap.add_argument('--solve', action='store_true')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--out', default=str(OUT_DIR))
    args = ap.parse_args()
    if not (args.solve or args.check):
        ap.error('need --solve or --check')
    out_dir = Path(args.out)
    if args.check and not args.solve:
        audit_path = out_dir / 'independent_audit.json'
        if not audit_path.exists():
            print(f'no audit at {audit_path}; run --solve first')
            return 2
        audit_json = json.loads(audit_path.read_text(encoding='utf-8'))
        print(json.dumps({k: audit_json[k] for k in
                          ('candidate_solved', 'independent_audit_passed',
                           'user_adopted', 'optimizer_fully_met',
                           'hard_acceptance', 'counts')}, ensure_ascii=False, indent=2))
        return 0 if audit_json['independent_audit_passed'] else 1
    res = run_annual(out_dir)
    print(f'M0 annual: plan {res["plan"]:.6f} yuan, emg {res["emg"]:.6f} yuan, '
          f'total {res["total"]:.6f} yuan, jan {res["jan"]:.6f} yuan, '
          f'jan+total {res["jan_plus_total"]:.6f} yuan')
    print(f'fallbacks {res["fallbacks"]}, year-end E {res["e_end"]:.6f} kWh, '
          f'wall {res["wall_s"]:.1f} s')
    print('independent_audit_passed = '
          f'{res["audit"]["independent_audit_passed"]} '
          f'(cons residual {res["audit"]["hard_acceptance"]["conservation_residual_max_kwh"]:.3g}, '
          f'E residual {res["audit"]["hard_acceptance"]["E_recursion_residual_max_kwh"]:.3g}, '
          f'cost diff {res["audit"]["hard_acceptance"]["cost_recompute_diff_yuan"]:.3g})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
