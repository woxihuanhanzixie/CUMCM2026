#!/usr/bin/env python3
"""Q4-3 annual runner: four branches C0/C1/C2-P1 + C2-P7 (E1-LP10 control).

Usage:
  python Q4/run_q43.py --smoke     # four branches, Jan1-2 two-day smoke
  python Q4/run_q43.py --annual    # four branches, full year 0..52560
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from q43_runtime import Budget, load_inputs, Bases, Prices  # noqa: E402
from q43_controllers import make_branch  # noqa: E402

OUT = HERE / 'outputs' / 'Q43'
BRANCHES = [('C0', 'P1'), ('C1', 'P1'), ('C2', 'P1'), ('C2', 'P7')]
LIMITS = {'calls': 38000, 'fits': 2250, 'wall_seconds': 1200}


def run_branch(branch, data, budget, identity, bases, prices, end_day):
    obj = make_branch(branch, data, budget, identity, bases, prices)
    jan = feb = jan_g = feb_g = 0.0
    rows = []
    for day in range(end_day):
        while obj.position < (day + 1) * 144:
            row = obj.step()
            rows.append(row)
            if day < 31:
                jan += row['total']
                jan_g += row['g']
            else:
                feb += row['total']
                feb_g += row['g']
    return dict(name=branch[0], method=branch[1], jan=jan, feb_dec=feb,
                jan_g=jan_g, feb_g=feb_g, final_soc=obj.energy, rows=rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--annual', action='store_true')
    args = ap.parse_args()
    if not (args.smoke or args.annual):
        ap.error('need --smoke or --annual')
    OUT.mkdir(parents=True, exist_ok=True)

    budget = Budget(LIMITS)
    data = load_inputs()
    identity = {'model': 'Q4-3-E1LP10', 'input_sha256': data.pop('input_sha256')}
    bases = Bases(budget)
    prices = Prices(budget)
    end_day = 2 if args.smoke else 365

    results = []
    t0 = time.perf_counter()
    for branch in BRANCHES:
        t = time.perf_counter()
        r = run_branch(branch, data, budget, identity, bases, prices, end_day)
        r['wall'] = time.perf_counter() - t
        results.append(r)
        print(f"[{r['wall']:6.1f}s] {branch[0]}-{branch[1]}  "
              f"jan={r['jan']:,.2f}  feb_dec={r['feb_dec']:,.2f}  "
              f"feb_g={r['feb_g']:,.0f}  soc={r['final_soc']:.1f}", flush=True)

    summary = {r['name'] + '-' + r['method']:
               {k: r[k] for k in ('jan', 'feb_dec', 'jan_g', 'feb_g', 'final_soc', 'wall')}
               for r in results}
    summary['budget'] = budget.record()
    summary['elapsed'] = time.perf_counter() - t0

    if args.smoke:
        out = OUT / 'smoke_summary.json'
    else:
        out = OUT / 'annual_summary.json'
        j = {r['name'] + '-' + r['method']: r for r in results}
        dC0 = j['C0-P1']
        dC1 = j['C1-P1']
        dC2 = j['C2-P1']
        dP7 = j['C2-P7']
        summary['delta_info'] = dC0['feb_dec'] - dC1['feb_dec']
        summary['delta_adjust'] = dC1['feb_dec'] - dC2['feb_dec']
        summary['delta_price'] = dC2['feb_dec'] - dP7['feb_dec']
        # monthly + full ledger CSV
        rows = []
        for r in results:
            for row in r['rows']:
                rows.append(row)
        import csv as _csv
        with (OUT / 'ledger.csv').open('w', encoding='utf-8', newline='') as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with (OUT / 'price_coefficients.json').open('w', encoding='utf-8') as f:
            json.dump(prices.coeffs, f, ensure_ascii=False, indent=2)

    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                      for k, v in summary.items() if k != 'budget'},
                     ensure_ascii=False))
    print(f'budget: {budget.record()}')
    print(f'output: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
