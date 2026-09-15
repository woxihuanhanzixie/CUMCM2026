#!/usr/bin/env python3
"""Q4-3 post-hoc summaries: run_manifest.json + monthly_summary.csv (from ledger)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs' / 'Q43'
LEDGER = OUT / 'ledger.csv'


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    df = pd.read_csv(LEDGER)

    # ---- monthly summary ----
    df['ym'] = (pd.Timestamp('2025-01-01') + pd.to_timedelta(df['position'] * 10, unit='m')).dt.strftime('%Y-%m')
    rows = []
    for (pol, meth), g in df.groupby(['policy', 'method'], sort=True):
        for ym, m in g.groupby('ym'):
            rows.append(dict(policy=pol, method=meth, month=ym,
                             normal=float(m['normal'].sum()),
                             adjustment=float(m['adjustment'].sum()),
                             emergency=float(m['emergency'].sum()),
                             total=float(m['total'].sum()),
                             emg_kwh=float(m['g'].sum()),
                             waste_kwh=float(m['w'].sum()),
                             charge_kwh=float(m['c'].sum()),
                             discharge_kwh=float(m['s'].sum()),
                             E_start=float(m['E_before'].iloc[0]),
                             E_end=float(m['E_after'].iloc[-1])))
    monthly = pd.DataFrame(rows)
    monthly.to_csv(OUT / 'monthly_summary.csv', index=False)

    # ---- run manifest ----
    vendor = HERE.parent / 'Q3' / 'teammate_e1lp10' / 'implementation'
    code = {p.name: sha(p) for p in [HERE / 'run_q43.py', HERE / 'q43_runtime.py',
                                     HERE / 'q43_controllers.py', HERE / 'audit_q43.py',
                                     HERE / 'export_result43.py']}
    code.update({f'vendor/{p.name}': sha(p) for p in
                 [vendor / 'core.py', vendor / 'controller.py',
                  vendor / 'forecasting.py', vendor / 'data_access.py']})
    annual = json.loads((OUT / 'annual_summary.json').read_text(encoding='utf-8'))
    manifest = dict(
        question='Q4-3', strategy='C0/C1/C2-P1 + C2-P7 (E1-LP10 hourly re-plan)',
        price_method='P1 weekly-diff OLS+AR(1); P7 last-week same-slot; P0 short-history fallback',
        model_sha256=code,
        input_sha256=annual.get('input_sha256', {}),
        environment=dict(python=sys.version.split()[0], numpy=np.__version__),
        mapping_id='Q43', timezone='Asia/Shanghai', delta_h=1 / 6,
        start=0, end=52560, initial_state='Jan1 E=6000 per branch',
        budget=annual['budget'],
        run_status='PASS_ANNUAL', formal_workbook='result4-3_DRAFT.xlsx',
        results={k: v for k, v in annual.items() if k not in ('budget', 'elapsed')},
    )
    (OUT / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                           encoding='utf-8')
    print('monthly_summary.csv:', len(monthly), 'rows')
    print('run_manifest.json written')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
