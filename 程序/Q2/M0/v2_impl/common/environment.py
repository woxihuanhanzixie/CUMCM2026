#!/usr/bin/env python3
"""Q2 v2 shared physics: battery environment and frozen greedy executor.

Frozen greedy (spec 01): r = q - n (n = actual net kWh per slot);
c = min(r+, C, (E_HI-e)/eta_c); s = min((-r)+, C, eta_d*(e-E_LO));
z = (-r-s)+; w = (r-c)+; e <- e + eta_c*c - s/eta_d.
"""

from __future__ import annotations

import numpy as np

from .data_io import DT, ETA_C, ETA_D, E_LO, E_HI, E_PORT_MAX, EMERGENCY_MULT, T


def greedy_step(e: float, net: float, q: float) -> dict[str, float]:
    r = q - net
    c = min(max(r, 0.0), E_PORT_MAX, (E_HI - e) / ETA_C)
    s = min(max(-r, 0.0), E_PORT_MAX, ETA_D * (e - E_LO))
    c = max(0.0, c)
    s = max(0.0, s)
    z = max(-r - s, 0.0)
    w = max(r - c, 0.0)
    return {'c': c, 's': s, 'z': z, 'w': w, 'e': e + ETA_C * c - s / ETA_D}


def simulate_day(net, q, e0, price=None) -> dict[str, np.ndarray | float]:
    net = np.asarray(net, float)
    q = np.asarray(q, float)
    if net.shape != (T,) or q.shape != (T,):
        raise ValueError(f'simulate_day expects ({T},) arrays, got {net.shape}/{q.shape}')
    c = np.zeros(T)
    s = np.zeros(T)
    z = np.zeros(T)
    w = np.zeros(T)
    E = np.empty(T)
    e = float(e0)
    for t in range(T):
        step = greedy_step(e, float(net[t]), float(q[t]))
        c[t], s[t], z[t], w[t] = step['c'], step['s'], step['z'], step['w']
        e = step['e']
        E[t] = e
    out: dict[str, np.ndarray | float] = {'c': c, 's': s, 'z': z, 'w': w, 'E': E,
                                          'e_end': float(e), 'e0': float(e0)}
    if price is not None:
        out['emg_cost'] = float(EMERGENCY_MULT * np.dot(np.asarray(price, float), z))
    return out


def check_physics(z, c, s, e0, e_end, tol: float = 1e-7) -> list[str]:
    """Hard physical bounds that every model output must satisfy."""
    errs: list[str] = []
    if np.min(z) < -tol:
        errs.append(f'negative emergency z: min {np.min(z):.3g}')
    if np.max(c) > E_PORT_MAX + tol or np.max(s) > E_PORT_MAX + tol:
        errs.append(f'port power bound violated: max c/s {max(np.max(c), np.max(s)):.3g}')
    if np.min(e0) < E_LO - tol or np.max(e_end) > E_HI + tol:
        errs.append(f'soc bounds violated: e0 min {np.min(e0):.3g}, e_end max {np.max(e_end):.3g}')
    if np.max(np.asarray(c) * np.asarray(z)) > tol:
        errs.append('simultaneous charge and emergency purchase')
    if np.max(np.asarray(c) * np.asarray(s)) > tol:
        errs.append('simultaneous charge and discharge')
    return errs
