#!/usr/bin/env python3
"""Q2 v2 independent ledger: recompute dispatch consequences from raw actions.

The ledger is a second, deliberately simple implementation of the energy-flow
identity (supply = q - c + s; z = (n - supply)+; w = (supply - n)+) used to
audit model outputs, so no model can claim favorable numbers without them
being recomputed.
"""

from __future__ import annotations

import numpy as np

from .data_io import ETA_C, ETA_D, E_LO, E_HI, E_PORT_MAX, EMERGENCY_MULT

FLOW_TOL = 1e-6   # kWh, per slot/state
COST_TOL = 0.01   # yuan


def recompute(net, q, c, s, e0) -> dict[str, np.ndarray | float]:
    net = np.asarray(net, float)
    q = np.asarray(q, float)
    c = np.asarray(c, float)
    s = np.asarray(s, float)
    supply = q - c + s
    z = np.maximum(net - supply, 0.0)
    w = np.maximum(supply - net, 0.0)
    E = np.empty(net.shape[0])
    e = float(e0)
    for t in range(net.shape[0]):
        e += ETA_C * c[t] - s[t] / ETA_D
        E[t] = e
    return {'z': z, 'w': w, 'E': E, 'e_end': float(e)}


def audit(net, q, c, s, e0, claimed, price=None) -> tuple[list[str], dict]:
    """Recompute and compare against claimed z/w/E/e_end/emg_cost; also check physics."""
    r = recompute(net, q, c, s, e0)
    errs: list[str] = []
    for key in ('z', 'w', 'E'):
        if claimed.get(key) is not None:
            diff = np.max(np.abs(np.asarray(claimed[key], float) - r[key]))
            if diff > FLOW_TOL:
                errs.append(f'ledger mismatch {key}: max |diff| = {diff:.3g}')
    if claimed.get('e_end') is not None and abs(float(claimed['e_end']) - r['e_end']) > FLOW_TOL:
        errs.append(f"ledger mismatch e_end: claimed {claimed['e_end']} vs {r['e_end']:.6g}")
    if price is not None and claimed.get('emg_cost') is not None:
        rc = float(EMERGENCY_MULT * np.dot(np.asarray(price, float), r['z']))
        if abs(rc - float(claimed['emg_cost'])) > COST_TOL:
            errs.append(f"ledger cost mismatch: claimed {claimed['emg_cost']} vs recomputed {rc}")
    if np.min(r['z']) < -FLOW_TOL or np.min(r['w']) < -FLOW_TOL:
        errs.append('ledger recomputed negative z/w')
    if np.max(c) > E_PORT_MAX + 1e-7 or np.max(s) > E_PORT_MAX + 1e-7:
        errs.append('ledger power bounds violated')
    if np.min(r['E']) < E_LO - 1e-6 or np.max(r['E']) > E_HI + 1e-6:
        errs.append('ledger soc bounds violated')
    if np.max(c * s) > 1e-7:
        errs.append('ledger simultaneous charge/discharge')
    if np.max(c * r['z']) > 1e-7:
        errs.append('ledger simultaneous charge/emergency')
    return errs, r
