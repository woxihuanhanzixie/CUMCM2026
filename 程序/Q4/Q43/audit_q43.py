#!/usr/bin/env python3
"""Q4-3 independent audit: physics, E recursion, bounds, fee recompute from ledger."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LEDGER = HERE / 'outputs' / 'Q43' / 'ledger.csv'
OUT = HERE / 'outputs' / 'Q43' / 'independent_audit.json'

ETA = 0.9
LO, HI = 1200.0, 10800.0
PORT = 5000.0 / 6
TOL_E = 1e-6
TOL_FEE = 1e-3


def audit_branch(df: pd.DataFrame, name: str) -> dict:
    d = df.sort_values('position').reset_index(drop=True)
    a = d['a_final'].values
    q = d['q0'].values
    ell = d['ell'].values
    pv = d['pv'].values
    price = d['price'].values
    c = d['c'].values
    s = d['s'].values
    g = d['g'].values
    w = d['w'].values
    eb = d['E_before'].values
    ea = d['E_after'].values

    cons = a + pv + s + g - ell - c - w
    erec = ea - eb - ETA * c + s / ETA
    e_cont = np.abs(ea[:-1] - eb[1:]).max()
    cs = (c * s).max()
    cg = (c * g).max()
    e_min, e_max = ea.min(), ea.max()
    c_max, s_max = c.max(), s.max()
    g_min, w_min = g.min(), w.min()

    fee = np.abs(price * a - d['normal'].values).max()
    fee = max(fee, np.abs(0.5 * price * np.abs(a - q) - d['adjustment'].values).max())
    fee = max(fee, np.abs(5 * price * g - d['emergency'].values).max())
    fee = max(fee, np.abs(d['normal'].values + d['adjustment'].values
                          + d['emergency'].values - d['total'].values).max())

    return {
        'branch': name,
        'segments': int(len(d)),
        'conservation_residual_max_kwh': float(np.abs(cons).max()),
        'E_recursion_residual_max_kwh': float(np.abs(erec).max()),
        'E_continuity_residual_max_kwh': float(e_cont),
        'simultaneous_charge_discharge_max': float(cs),
        'simultaneous_charge_emergency_max': float(cg),
        'E_min_kwh': float(e_min), 'E_max_kwh': float(e_max),
        'c_max_kwh': float(c_max), 's_max_kwh': float(s_max),
        'g_min_kwh': float(g_min), 'w_min_kwh': float(w_min),
        'fee_recompute_diff_max_yuan': float(fee),
        'bounds_ok': bool(e_min >= LO - 1e-6 and e_max <= HI + 1e-6
                          and c_max <= PORT + 1e-6 and s_max <= PORT + 1e-6
                          and g_min >= -1e-9 and w_min >= -1e-9),
        'physics_ok': bool(np.abs(cons).max() <= TOL_E and np.abs(erec).max() <= TOL_E
                           and cs <= 1e-6 and cg <= 1e-6),
        'fee_ok': bool(fee <= TOL_FEE),
    }


def main() -> int:
    df = pd.read_csv(LEDGER)
    branches = []
    for pol, meth in [('C0', 'P1'), ('C1', 'P1'), ('C2', 'P1'), ('C2', 'P7')]:
        m = df[(df['policy'] == pol) & (df['method'] == meth)]
        assert len(m) == 52560, f'{pol}-{meth} segments {len(m)} != 52560'
        branches.append(audit_branch(m, f'{pol}-{meth}'))

    # main-eval totals recomputed independently
    totals = {}
    for pol, meth in [('C0', 'P1'), ('C1', 'P1'), ('C2', 'P1'), ('C2', 'P7')]:
        m = df[(df['policy'] == pol) & (df['method'] == meth)]
        feb = m[m['position'] >= 31 * 144]
        jan = m[m['position'] < 31 * 144]
        totals[f'{pol}-{meth}'] = dict(jan=float(jan['total'].sum()),
                                       feb_dec=float(feb['total'].sum()),
                                       feb_g=float(feb['g'].sum()))

    report = {
        'model': 'Q4-3-E1LP10',
        'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'branches': branches,
        'totals': totals,
        'all_passed': bool(all(b['physics_ok'] and b['fee_ok'] and b['bounds_ok']
                               for b in branches)),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for b in branches:
        print(f"{b['branch']}: cons={b['conservation_residual_max_kwh']:.2e} "
              f"Erec={b['E_recursion_residual_max_kwh']:.2e} "
              f"Econt={b['E_continuity_residual_max_kwh']:.2e} "
              f"fee={b['fee_recompute_diff_max_yuan']:.2e} "
              f"E[{b['E_min_kwh']:.1f},{b['E_max_kwh']:.1f}] "
              f"cs={b['simultaneous_charge_discharge_max']:.2e} "
              f"cg={b['simultaneous_charge_emergency_max']:.2e}")
    print('all_passed =', report['all_passed'])
    print('output:', OUT)
    return 0 if report['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
