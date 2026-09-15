#!/usr/bin/env python3
"""Q2 v2 forecast archive export: k0 ridge load/PV/net per publish day.

The archive is shared by all three models (spec 05: 预测档案所有模型共用).
Export days Jan22 (index 21) .. Dec31 (index 364): each publish day h holds
the forecast made at midnight of day h for day h itself (k=0), using only
history strictly before h.

Usage:
  python Q2/v2_impl/export_forecast.py            # k0 -> outputs/forecast_archive/k0.csv
  python Q2/v2_impl/export_forecast.py --k 1      # k1 (day-ahead) if requested
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v2_impl.common.data_io import (DT, N_DAYS, Q2_ROOT, T,  # noqa: E402
                                    load_attachment1, load_attachment2)
from v2_impl.common.forecast import RidgeForecaster  # noqa: E402

POOL_START = 21
OUT_DIR = Q2_ROOT / 'outputs' / 'forecast_archive'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=0, choices=(0, 1, 2))
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    a1 = load_attachment1()
    a2 = load_attachment2()
    data = {'price': a1['price'], 'load_act': a2['load_act'],
            'pv_act': a2['pv_act'], 'dates': a2['dates']}
    fc = RidgeForecaster(data)
    out_dir = Path(args.out) if args.out else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'k{args.k}.csv'
    lo = max(POOL_START, 14 + args.k + 6)  # ensure training >= 7 days
    with path.open('w', encoding='utf-8', newline='') as f:
        f.write('publish_day_index,date,slot,load_hat_kw,pv_hat_kw,net_hat_kwh\n')
        for h in range(lo, N_DAYS - args.k):
            pr = fc.predict_day(h, args.k)
            load_h = np.clip(pr['load'], 0.0, None)
            pv_h = np.clip(pr['pv'], 0.0, None)
            net_h = (load_h - pv_h) * DT
            date = a2['dates'][h + args.k]
            for t in range(T):
                f.write(f'{h},{date.isoformat()},{t + 1},{load_h[t]:.12g},'
                        f'{pv_h[t]:.12g},{net_h[t]:.12g}\n')
    n_rows = (N_DAYS - args.k - lo) * T
    print(f'wrote {path} ({n_rows} rows, publish days {lo}..{N_DAYS - args.k - 1})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
