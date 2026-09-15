#!/usr/bin/env python3
"""Q2 v2 January initialization (frozen protocol convention, not an optimization).

Policy (protocol.json / spec 01):
  1/1: E = 6000, q = c = s = 0, z = n+, w = (-n)+;
  1/2..1/31: n_hat = mean of the preceding up-to-7 days' actual net;
             q = n_hat+; then frozen greedy execution; continuous recursion.
Reference to reproduce: Feb1 E = 10800, January total purchase cost
(plan at 1x + emergency at 5x) = 3130627.3752977448.
"""

from __future__ import annotations

import numpy as np

from .data_io import DT, E_INIT, EMERGENCY_MULT, T
from .environment import simulate_day

JAN_DAYS = 31
N_JAN_SLOTS = JAN_DAYS * T  # 4464
REF_JAN_COST = 3130627.3752977448
REF_FEB1_ENERGY = 10800.0
COST_TOL = 0.01
ENERGY_TOL = 1e-6


def run_january(price, load_act, pv_act) -> dict:
    net_act = (np.asarray(load_act, float) - np.asarray(pv_act, float)) * DT
    q = np.zeros((JAN_DAYS, T))
    c = np.zeros_like(q)
    s = np.zeros_like(q)
    z = np.zeros_like(q)
    w = np.zeros_like(q)
    E = np.zeros_like(q)
    e0 = np.zeros(JAN_DAYS)
    e_end = np.zeros(JAN_DAYS)

    e = E_INIT
    e0[0] = e
    z[0] = np.maximum(net_act[0], 0.0)
    w[0] = np.maximum(-net_act[0], 0.0)
    E[0] = e
    e_end[0] = e

    for d in range(1, JAN_DAYS):
        n_hat = net_act[max(0, d - 7):d].mean(axis=0)
        q[d] = np.maximum(n_hat, 0.0)
        e0[d] = e
        sim = simulate_day(net_act[d], q[d], e)
        c[d] = sim['c']
        s[d] = sim['s']
        z[d] = sim['z']
        w[d] = sim['w']
        E[d] = sim['E']
        e = sim['e_end']
        e_end[d] = e

    plan_cost = float(np.dot(np.asarray(price, float), q.sum(axis=0)))
    emg_cost = EMERGENCY_MULT * float(np.dot(np.asarray(price, float), z.sum(axis=0)))
    jan_cost = plan_cost + emg_cost
    return {'q': q, 'c': c, 's': s, 'z': z, 'w': w, 'E': E, 'e0': e0, 'e_end': e_end,
            'plan_cost': plan_cost, 'emg_cost': emg_cost, 'jan_cost': jan_cost,
            'feb1_energy': float(e_end[JAN_DAYS - 1])}


def check_january(res: dict) -> list[str]:
    errs: list[str] = []
    if abs(res['jan_cost'] - REF_JAN_COST) > COST_TOL:
        errs.append(f"january fee {res['jan_cost']:.10f} != reference {REF_JAN_COST}")
    if abs(res['feb1_energy'] - REF_FEB1_ENERGY) > ENERGY_TOL:
        errs.append(f"feb1 energy {res['feb1_energy']:.10f} != reference {REF_FEB1_ENERGY}")
    return errs
