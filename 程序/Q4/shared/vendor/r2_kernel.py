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
@lru_cache(maxsize=300)
def _lp_pattern(n: int, with_z: bool):
    """Equality pattern for a battery LP over n slots (sparse csc).

    Rows 0..n-1: balance Q - c + s [+ z] - w = nbar (rhs filled by caller).
    Rows n..2n-1: E_t - .9c_t + s_t/.9 - E_{t-1} = 0 (first row rhs = e0).
    Variable layout without z: [Q,c,s,w,E] (5n); with z: [Q,c,s,z,w,E] (6n).
    """
    if with_z:
        off = {'Q': 0, 'c': n, 's': 2 * n, 'z': 3 * n, 'w': 4 * n, 'E': 5 * n}
        nv = 6 * n
        rows = np.concatenate([np.arange(n)] * 5)
        cols = np.concatenate([np.arange(n), n + np.arange(n), 2 * n + np.arange(n), 3 * n + np.arange(n), 4 * n + np.arange(n)])
        data = np.concatenate([np.ones(n), -np.ones(n), np.ones(n), np.ones(n), -np.ones(n)])
    else:
        off = {'Q': 0, 'c': n, 's': 2 * n, 'w': 3 * n, 'E': 4 * n}
        nv = 5 * n
        rows = np.concatenate([np.arange(n)] * 4)
        cols = np.concatenate([np.arange(n), n + np.arange(n), 2 * n + np.arange(n), 3 * n + np.arange(n)])
        data = np.concatenate([np.ones(n), -np.ones(n), np.ones(n), -np.ones(n)])
    rows2 = np.concatenate([n + np.arange(n)] * 3 + [n + np.arange(1, n)])
    cols2 = np.concatenate([off['E'] + np.arange(n), off['c'] + np.arange(n), off['s'] + np.arange(n), off['E'] + np.arange(n - 1)])
    data2 = np.concatenate([np.ones(n), -ETA_C * np.ones(n), 1.0 / ETA_D * np.ones(n), -np.ones(max(n - 1, 0))])
    Aeq = coo_matrix((np.concatenate([data, data2]), (np.concatenate([rows, rows2]), np.concatenate([cols, cols2]))), shape=(2 * n, nv)).tocsc()
    return (Aeq, off)

def _validate_lp(nbar, p, e0, q_fixed=()):
    nbar, p, q = (np.asarray(nbar, float), np.asarray(p, float), np.asarray(q_fixed, float))
    if nbar.ndim != 1 or not len(nbar) or p.shape != nbar.shape:
        raise ValueError('invalid horizon arrays')
    if not all((np.isfinite(x).all() for x in (nbar, p, q))) or not np.isfinite(e0):
        raise ValueError('nonfinite input/state')
    if np.any(p <= 0) or np.any(q < -FEAS_TOL) or len(q) > len(nbar):
        raise ValueError('invalid price/locked order')
    if not E_LO - FEAS_TOL <= e0 <= E_HI + FEAS_TOL:
        raise ValueError('state outside physical bounds')

def build_midnight_lp(nbar, p, e0):
    """n-slot midnight LP (no z). Variables: Q, c, s, w, E (5n); 2n eq rows."""
    n = len(nbar)
    nbar = np.asarray(nbar, float)
    p = np.asarray(p, float)
    Aeq, off = _lp_pattern(n, False)
    beq = np.zeros(2 * n)
    beq[:n] = nbar
    beq[n] = float(e0)
    obj = np.zeros(5 * n)
    obj[:n] = p
    bounds = [(0.0, None)] * n + [(0.0, E_PORT_MAX)] * n + [(0.0, E_PORT_MAX)] * n + [(0.0, None)] * n + [(E_LO, E_HI)] * n
    return (obj, Aeq, beq, bounds, off)

def build_intraday_lp(nbar, p, e0, q_fixed):
    """Intraday LP: F slots (Q locked, z allowed, c capped by q-nbar) then
    U slots (Q free, z=0). Variables: Q, c, s, z, w, E (6n); 2n eq rows."""
    n = len(nbar)
    nF = len(q_fixed)
    nbar = np.asarray(nbar, float)
    p = np.asarray(p, float)
    q_fixed = np.asarray(q_fixed, float)
    Aeq, off = _lp_pattern(n, True)
    beq = np.zeros(2 * n)
    beq[:n] = nbar
    beq[n] = float(e0)
    obj = np.zeros(6 * n)
    obj[off['z']:off['z'] + nF] = EMERGENCY_MULT * p[:nF]
    obj[nF:n] = p[nF:n]
    c_ub = np.minimum(E_PORT_MAX, np.maximum(q_fixed - nbar[:nF], 0.0))
    bounds = [(float(q_fixed[i]), float(q_fixed[i])) for i in range(nF)] + [(0.0, None)] * (n - nF) + [(0.0, float(c_ub[i])) for i in range(nF)] + [(0.0, E_PORT_MAX)] * (n - nF) + [(0.0, E_PORT_MAX)] * n + [(0.0, None)] * nF + [(0.0, 0.0)] * (n - nF) + [(0.0, None)] * n + [(E_LO, E_HI)] * n
    if nF < 1:
        raise ValueError('intraday needs a locked current slot')
    surplus = max(float(q_fixed[0] - nbar[0]), 0.0)
    charge = min(surplus, E_PORT_MAX, max(E_HI - float(e0), 0.0) / ETA_C)
    bounds[off['c']] = (charge, charge)
    bounds[off['s']] = (0.0, min(E_PORT_MAX, max(float(nbar[0] - q_fixed[0]), 0.0), max(float(e0) - E_LO, 0.0) * ETA_D))
    return (obj, Aeq, beq, bounds, off, nF)

def _check_solution(nbar, p, e0, q_fixed, x, off, with_z, tol=FEAS_TOL):
    """Independent feasibility check of a raw LP solution."""
    n = len(nbar)
    Q = x[off['Q']:off['Q'] + n]
    c = x[off['c']:off['c'] + n]
    s = x[off['s']:off['s'] + n]
    z = x[off['z']:off['z'] + n] if with_z else np.zeros(n)
    w = x[off['w']:off['w'] + n]
    E = x[off['E']:off['E'] + n]
    bal = Q - c + s + z - w - np.asarray(nbar)
    eres = np.empty(n)
    eres[0] = E[0] - float(e0) - ETA_C * c[0] + s[0] / ETA_D
    eres[1:] = E[1:] - E[:-1] - ETA_C * c[1:] + s[1:] / ETA_D
    viol: list[str] = []
    worst = max(float(np.max(np.abs(bal))), float(np.max(np.abs(eres))))
    if worst > tol:
        viol.append(f'max_eq_residual {worst:.3g}')
    if np.min(Q) < -tol:
        viol.append('Q negative')
    if len(q_fixed):
        if np.max(np.abs(Q[:len(q_fixed)] - np.asarray(q_fixed))) > tol:
            viol.append('F Q not locked')
        nF = len(q_fixed)
        c_ub = np.minimum(E_PORT_MAX, np.maximum(np.asarray(q_fixed) - np.asarray(nbar[:nF]), 0.0))
        if np.any(c[:nF] > c_ub + tol):
            viol.append('F c above (q-nbar)+ cap')
        if np.any(z[nF:] > tol):
            viol.append('U z not fixed to 0')
        charge = min(float(c_ub[0]), max(E_HI - float(e0), 0.0) / ETA_C)
        discharge = min(E_PORT_MAX, max(float(nbar[0] - q_fixed[0]), 0.0), max(float(e0) - E_LO, 0.0) * ETA_D)
        if abs(c[0] - charge) > tol or s[0] > discharge + tol:
            viol.append('R2 observed-root action constraint')
    if np.min(c) < -tol or np.max(c) > E_PORT_MAX + tol:
        viol.append('c port bound')
    if np.min(s) < -tol or np.max(s) > E_PORT_MAX + tol:
        viol.append('s port bound')
    if np.min(z) < -tol:
        viol.append('z negative')
    if np.min(w) < -tol:
        viol.append('w negative')
    if np.min(E) < E_LO - tol or np.max(E) > E_HI + tol:
        viol.append('E bound')
    return (viol, worst, (Q, c, s, z, w, E))

def solve_midnight_lp(nbar, p, e0, time_limit=TIME_LIMIT):
    _validate_lp(nbar, p, e0)
    obj, Aeq, beq, bounds, off = build_midnight_lp(nbar, p, e0)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', OptimizeWarning)
        res = linprog(obj, A_eq=Aeq, b_eq=beq, bounds=bounds, method='highs', options={'primal_feasibility_tolerance': SOLVE_TOL, 'dual_feasibility_tolerance': SOLVE_TOL, 'time_limit': time_limit, 'random_seed': SEED})
    wall = time.perf_counter() - t0
    out = {'status': int(res.status), 'status_text': str(res.message).strip(), 'wall_s': wall, 'nit': getattr(res, 'nit', None), 'usable': False, 'max_residual': None, 'H': len(nbar)}
    if res.status in (0, 1) and res.x is not None:
        viol, worst, parts = _check_solution(nbar, p, e0, [], res.x, off, False)
        out['max_residual'] = worst
        if not viol:
            out['usable'] = True
            out['Q'], out['c'], out['s'], _z, out['w'], out['E'] = parts
            out['fun'] = float(np.dot(obj, res.x))
        else:
            out['violations'] = viol
    return out

def solve_intraday_lp(nbar, p, e0, q_fixed, time_limit=TIME_LIMIT):
    _validate_lp(nbar, p, e0, q_fixed)
    obj, Aeq, beq, bounds, off, nF = build_intraday_lp(nbar, p, e0, q_fixed)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', OptimizeWarning)
        res = linprog(obj, A_eq=Aeq, b_eq=beq, bounds=bounds, method='highs', options={'primal_feasibility_tolerance': SOLVE_TOL, 'dual_feasibility_tolerance': SOLVE_TOL, 'time_limit': time_limit, 'random_seed': SEED})
    out = {'status': int(res.status), 'status_text': str(res.message).strip(), 'nit': getattr(res, 'nit', None), 'usable': False, 'max_residual': None, 'H': len(nbar), 'nF': nF, 'solver_calls': 1, 'secondary': {'attempted': False, 'accepted': False}, 'primary_optimal': res.status == 0}
    if res.status in (0, 1) and res.x is not None:
        viol, worst, parts = _check_solution(nbar, p, e0, q_fixed, res.x, off, True)
        out['max_residual'] = worst
        if not viol:
            out['usable'] = True
            primary_fun = float(np.dot(obj, res.x))
            out['primary_fun'] = primary_fun
            out['primary_lp'] = dict(zip(('Q', 'c', 's', 'z', 'w', 'E'), parts))
            chosen = res.x
            remaining = time_limit - (time.perf_counter() - t0)
            if remaining > 0.001:
                secondary_obj = np.zeros_like(obj)
                secondary_obj[off['E']] = -1.0
                out['solver_calls'] += 1
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', OptimizeWarning)
                        second = linprog(secondary_obj, A_eq=Aeq, b_eq=beq, A_ub=csr_matrix(obj.reshape(1, -1)), b_ub=[primary_fun + LEX_COST_EPS], bounds=bounds, method='highs', options={'primal_feasibility_tolerance': SOLVE_TOL, 'dual_feasibility_tolerance': SOLVE_TOL, 'time_limit': remaining, 'random_seed': SEED})
                    sec = {'attempted': True, 'status': int(second.status), 'message': str(second.message), 'accepted': False, 'optimal': second.status == 0, 'cost_cap_yuan': primary_fun + LEX_COST_EPS}
                    if second.status in (0, 1) and second.x is not None:
                        v2, w2, _ = _check_solution(nbar, p, e0, q_fixed, second.x, off, True)
                        sec['main_cost_yuan'] = float(obj @ second.x)
                        sec['max_residual'] = w2
                        sec['root_energy_gain_kwh'] = float(second.x[off['E']] - res.x[off['E']])
                        if not v2 and sec['main_cost_yuan'] <= primary_fun + LEX_COST_EPS + SOLVE_TOL and (sec['root_energy_gain_kwh'] >= -FEAS_TOL):
                            chosen = second.x
                            sec['accepted'] = True
                        else:
                            sec['rejection'] = v2 or ['cost cap or root energy check failed']
                    out['secondary'] = sec
                except Exception as exc:
                    out['secondary'] = {'attempted': True, 'accepted': False, 'error': f'{type(exc).__name__}: {exc}'}
            else:
                out['secondary']['reason'] = 'primary used decision time budget'
            _, worst, parts = _check_solution(nbar, p, e0, q_fixed, chosen, off, True)
            out['max_residual'] = worst
            out['Q'], out['c'], out['s'], out['z'], out['w'], out['E'] = parts
            out['fun'] = float(np.dot(obj, chosen))
        else:
            out['violations'] = viol
    out['wall_s'] = time.perf_counter() - t0
    return out

def restore_solution(nbar, p, q_fixed, Q, c, s, z, w, E):
    """spec 03 §4 LP recovery: remove simultaneous c*s (delta transform, with
    ETA_C*ETA_D = 0.81 <= 1) then simultaneous z*w; re-verify feasibility,
    physical-mode flags and objective monotonicity."""
    n = len(nbar)
    nF = len(q_fixed)
    nbar = np.asarray(nbar, float)
    p = np.asarray(p, float)
    eta_prod = ETA_C * ETA_D
    delta = np.minimum(c, s / eta_prod)
    c_r = c - delta
    s_r = s - eta_prod * delta
    w_r = w + (1.0 - eta_prod) * delta
    m = np.minimum(z, w_r)
    z_r = z - m
    w_r = w_r - m
    for arr in (c_r, s_r, z_r, w_r):
        arr[np.abs(arr) < 1e-09] = 0.0
    bal = Q - c_r + s_r + z_r - w_r - nbar
    eres = ETA_C * c_r - s_r / ETA_D - (ETA_C * c - s / ETA_D)
    feas = float(np.max(np.abs(bal))) <= FEAS_TOL and float(np.max(np.abs(eres))) <= FEAS_TOL and (float(np.min(c_r)) >= -FEAS_TOL) and (float(np.max(c_r)) <= E_PORT_MAX + FEAS_TOL) and (float(np.min(s_r)) >= -FEAS_TOL) and (float(np.max(s_r)) <= E_PORT_MAX + FEAS_TOL) and (float(np.min(z_r)) >= -FEAS_TOL) and (float(np.min(w_r)) >= -FEAS_TOL) and (float(np.min(E)) >= E_LO - FEAS_TOL) and (float(np.max(E)) <= E_HI + FEAS_TOL)
    cs_ok = not bool(np.max(np.minimum(c_r, s_r)) > FEAS_TOL)
    zw_ok = not bool(np.max(np.minimum(z_r, w_r)) > FEAS_TOL)
    czF_ok = nF == 0 or not bool(np.max(np.minimum(c_r[:nF], z_r[:nF])) > FEAS_TOL)
    obj_raw = float(EMERGENCY_MULT * np.dot(p[:nF], z[:nF]) + np.dot(p[nF:], Q[nF:]))
    obj_r = float(EMERGENCY_MULT * np.dot(p[:nF], z_r[:nF]) + np.dot(p[nF:], Q[nF:]))
    return {'c': c_r, 's': s_r, 'z': z_r, 'w': w_r, 'E': E, 'feas_ok': feas, 'cs_ok': cs_ok, 'zw_ok': zw_ok, 'czF_ok': czF_ok, 'obj_raw': obj_raw, 'obj_restored': obj_r, 'obj_not_increased': obj_r <= obj_raw + 1e-06, 'delta_max': float(np.max(np.abs(delta))), 'zw_common_max': float(np.max(m))}

def fallback_q(n_hat):
    return np.maximum(np.asarray(n_hat, float), 0.0)
