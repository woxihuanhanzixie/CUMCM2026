from __future__ import annotations
import numpy as np
import time,warnings
from functools import lru_cache
from scipy.optimize import linprog,OptimizeWarning
from scipy.sparse import coo_matrix,csr_matrix
ETA_C=ETA_D=.9
E_LO=1200.; E_HI=10800.; E_PORT_MAX=5000/6
EMERGENCY_MULT=5; FEAS_TOL=1e-6; SOLVE_TOL=1e-7
TIME_LIMIT=10.; SEED=20260911; LEX_COST_EPS=1e-6; ALPHA=.8
def greedy_step(e: float, net: float, q: float) -> dict[str, float]:
    r = q - net
    c = min(max(r, 0.0), E_PORT_MAX, (E_HI - e) / ETA_C)
    s = min(max(-r, 0.0), E_PORT_MAX, ETA_D * (e - E_LO))
    c = max(0.0, c)
    s = max(0.0, s)
    z = max(-r - s, 0.0)
    w = max(r - c, 0.0)
    return {'c': c, 's': s, 'z': z, 'w': w, 'e': e + ETA_C * c - s / ETA_D}
