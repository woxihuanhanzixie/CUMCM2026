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
    bounds = [(0.0, None)] * n + [(0.0, E_PORT_MAX)] * n + [(0.0, E_PORT_MAX)] * n + [(0.0, float(wcap[t])) for t in range(n)] + [(E_LO, E_HI)] * n
    return (obj, Aeq, beq, bounds, off)

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
    return (viol, worst, (q, c, s, w, E))

def solve_plan_lp(a, p, wcap, e0, time_limit=TIME_LIMIT):
    """Solve the plan LP; return usable solution only if independently feasible."""
    obj, Aeq, beq, bounds, off = build_plan_lp(a, p, wcap, e0)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', OptimizeWarning)
        res = linprog(obj, A_eq=Aeq, b_eq=beq, bounds=bounds, method='highs', options={'primal_feasibility_tolerance': SOLVE_TOL, 'dual_feasibility_tolerance': SOLVE_TOL, 'time_limit': time_limit, 'random_seed': SEED})
    wall = time.perf_counter() - t0
    out = {'status': int(res.status), 'status_text': str(res.message).strip(), 'wall_s': wall, 'nit': getattr(res, 'nit', None), 'usable': False, 'max_residual': None}
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
