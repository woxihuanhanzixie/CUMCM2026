"""H48CT3: causal binary tree and sparse physical MILP, Phi=0.

Reduced formulation removes charge/discharge mode a by a state-preserving
cycle cancellation. Fixed-order nodes additionally impose c <= (q-net)+,
so their emergency mode b is unnecessary. Free-order nodes retain binary b.
The original formulation remains available for independent regression.
"""
from __future__ import annotations

import runtime  # noqa: F401
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

T = 144
PORT = 5000 / 6


def fit_params(blocks, decision_day):
    """blocks: (publish_day, actual-minus-original-paired-forecast[288])."""
    if len({h for h, _ in blocks}) != len(blocks):
        raise ValueError('duplicate publish day')
    if any(h < 21 or h + 1 >= decision_day for h, _ in blocks):
        raise ValueError('incomplete/future two-day residual block')
    blocks = sorted(blocks, key=lambda x: x[0])[-28:]
    if len(blocks) < 4:
        raise ValueError('model_input_failure: fewer than four complete blocks')
    e = np.asarray([row for _, row in blocks], float)
    if e.shape != (len(blocks), 288) or not np.isfinite(e).all():
        raise ValueError('expected finite paired residuals (blocks,288)')
    sigma = np.array([max(1., np.sqrt(np.mean(e[:, max(0,u-2):min(288,u+3)]**2)))
                      for u in range(288)])
    r = e / sigma
    raw = float(np.sum(r[:, :-1]*r[:, 1:]) / (np.sum(r[:, :-1]**2)+1e-6))
    phi = float(np.clip(raw, -.98, .98))
    kappa = float(np.sqrt(np.mean((r[:, 1:]-phi*r[:, :-1])**2)))
    return dict(sigma=sigma.tolist(), phi=phi, kappa=kappa,
                pool_days=[h for h, _ in blocks], n_blocks=len(blocks),
                raw_phi_ratio=raw, phi_was_clipped=abs(raw)>.98,
                mean_e=float(e.mean()), mean_standardized_residual=float(r.mean()))


def innovation_std(phi, kappa, steps):
    # At phi=0 this sum equals 1, not steps. Use the exact geometric sum.
    return kappa*np.sqrt(max(0., (1-phi**(2*steps))/(1-phi**2))) if steps else 0.


def build_tree(base, params, start=0, observed=None, ct4=False):
    """Nodes are post-observation actions; next-day q precedes u144 innovation."""
    base = np.asarray(base, float)
    H = len(base)
    sigma = np.asarray(params['sigma'], float)[:H]
    phi, kappa = params['phi'], params['kappa']
    if H not in (144, 288) or not 0 <= start < H:
        raise ValueError('invalid production horizon/start')
    if (sigma.shape != base.shape or not np.isfinite(base).all()
            or not np.isfinite(sigma).all() or (sigma < 1).any()
            or not -.98 <= phi <= .98 or not np.isfinite(kappa) or kappa < 0
            or (observed is not None and not np.isfinite(observed))):
        raise ValueError('invalid tree inputs')
    first = start + (observed is not None)
    points = {first}
    if first <= 144 < H:
        points.add(144)
    noon = next((u for u in (72,216) if first <= u < H), None)
    if noon is not None:
        points.add(noon)
    if ct4 and first <= 216 < H:
        points.add(216)
    points = sorted(u for u in points if u < H) if kappa else []
    # parent id, probability, history, last innovation time/value, tomorrow group
    live = [(-1, 1., (), start-1, 0., None)]
    nodes = []
    for u in range(start, H):
        nxt = []
        for parent, prob, hist, a, ra, tomorrow in live:
            if u == 144:
                tomorrow = ('tomorrow', hist)  # create order BEFORE innovation
            if observed is not None and u == start:
                alts = [(0, 1., (observed-base[u])/sigma[u], hist, u)]
            elif u in points:
                mu, std = phi**(u-a)*ra, innovation_std(phi,kappa,u-a)
                alts = [(s,.5,mu+s*std,hist+((u,s),),u) for s in (-1,1)]
            else:
                alts = [(0,1.,phi**(u-a)*ra,hist,a)]
            for sign, weight, r, history, anchor in alts:
                updated = (observed is not None and u == start) or u in points
                group = ('today',) if u < 144 else tomorrow
                if group is None:
                    raise ValueError('missing tomorrow order history')
                node = dict(id=len(nodes), parent=parent, slot=u, prob=prob*weight,
                            history=history, net=float(observed) if observed is not None
                            and u==start else float(base[u]+sigma[u]*r), order_key=group)
                nodes.append(node)
                nxt.append((node['id'],node['prob'],history,anchor,r if updated else ra,tomorrow))
        live = nxt
    return dict(nodes=nodes, leaves=[x[0] for x in live], branches=points,
                horizon=H, start=start, observed=observed, ct4=ct4)


@dataclass
class Solution:
    usable: bool
    actions: dict | None
    orders: dict
    log: dict


def solve(nodes, prices, energy, fixed=None, formulation='reduced', time_limit=5.,
          gap=1e-4, port=PORT):
    begin = perf_counter()
    fixed = {} if fixed is None else fixed
    if formulation not in ('original', 'reduced'):
        raise ValueError(formulation)
    if not nodes or not np.isfinite(energy) or not 1200 <= energy <= 10800:
        raise ValueError('invalid root energy/nodes')
    K = len(nodes)
    if [n['id'] for n in nodes] != list(range(K)):
        raise ValueError('node ids must be topological and contiguous')
    keys = list(dict.fromkeys((n['order_key'],n['slot']) for n in nodes))
    lookup = {key:j for j,key in enumerate(keys)}
    J = len(keys)
    qi = np.array([lookup[n['order_key'],n['slot']] for n in nodes])
    parents = np.array([n['parent'] for n in nodes])
    idx = np.arange(K)
    if np.any(parents >= idx) or np.any(parents < -1):
        raise ValueError('invalid parents')
    net = np.array([n['net'] for n in nodes])
    prob = np.array([n['prob'] for n in nodes])
    p = np.array([prices[n['slot']] for n in nodes])
    if not np.isfinite(net).all() or not np.isfinite(p).all() or np.any(p<=0):
        raise ValueError('invalid demands/prices')
    isfixed = np.array([key in fixed for key in keys])
    fixed_node = isfixed[qi]
    free_idx = np.flatnonzero(~fixed_node)
    original = formulation == 'original'
    nb = K if original else len(free_idx)
    nv = J + 5*K + nb + (K if original else 0)
    q = np.arange(J)
    c,s,z,w,e = [J+i*K+idx for i in range(5)]
    b = J+5*K+np.arange(nb)
    a = J+5*K+nb+idx if original else None
    cost=np.zeros(nv);lb=np.zeros(nv);ub=np.full(nv,np.inf);integer=np.zeros(nv)
    np.add.at(cost, qi[~fixed_node], (prob*p)[~fixed_node])
    cost[z]=5*prob*p
    ub[c]=ub[s]=port;ub[z]=np.maximum(net,0);lb[e]=1200;ub[e]=10800
    for key,value in fixed.items():
        if key in lookup:
            if not np.isfinite(value) or value < 0:
                raise ValueError('invalid fixed order')
            lb[lookup[key]]=ub[lookup[key]]=value
    ub[b]=1;integer[b]=1
    if original:
        ub[a]=1;integer[a]=1
    else:
        fi=np.flatnonzero(fixed_node)
        ub[c[fi]]=np.minimum(port,np.maximum(lb[qi[fi]]-net[fi],0))
    rr=[];cc=[];vv=[];lower=[];upper=[];row_count=0
    def add(columns, values, lo, hi):
        nonlocal row_count
        size=len(columns[0]);rows=row_count+np.arange(size)
        for col,val in zip(columns,values):
            rr.append(rows);cc.append(np.asarray(col));vv.append(np.broadcast_to(val,size))
        lower.extend(np.broadcast_to(lo,size));upper.extend(np.broadcast_to(hi,size))
        row_count += size
    add([qi,c,s,z,w],[1,-1,1,1,-1],net,net)
    rhs=np.where(parents<0,energy,0.)
    add([e,c,s],[1,-.9,1/.9],rhs,rhs)
    children=np.flatnonzero(parents>=0)
    rr.append(K+children);cc.append(e[parents[children]]);vv.append(-np.ones(len(children)))
    if original:
        add([c,a],[1,-port],-np.inf,0.)
        add([s,a],[1,port],-np.inf,port)
        bi=idx
    else:
        bi=free_idx
    if len(bi):
        add([z[bi],b],[1,-np.maximum(net[bi],0)],-np.inf,0.)
        add([c[bi],b],[1,port],-np.inf,port)
    matrix=coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),
                      shape=(row_count,nv)).tocsc()
    lower=np.asarray(lower);upper=np.asarray(upper)
    built=perf_counter()
    res=milp(cost,integrality=integer,bounds=Bounds(lb,ub),
             constraints=LinearConstraint(matrix,lower,upper),
             options={'time_limit':time_limit,'mip_rel_gap':gap,'presolve':True})
    ended=perf_counter()
    log=dict(status=int(res.status),message=str(res.message),solver_success=bool(res.success),
             objective=None if res.fun is None else float(res.fun),
             mip_gap=None if getattr(res,'mip_gap',None) is None else float(res.mip_gap),
             dual_bound=None if getattr(res,'mip_dual_bound',None) is None else float(res.mip_dual_bound),
             mip_nodes=getattr(res,'mip_node_count',None), build_seconds=built-begin,
             solve_seconds=ended-built,nodes=K,variables=nv,binaries=int(integer.sum()),
             formulation=formulation,usable=False)
    if res.x is None or not np.isfinite(res.x).all():
        return Solution(False,None,{},log)
    x=res.x
    ax=matrix@x
    residual=max(float(np.max(np.maximum(lower-ax,0))),float(np.max(np.maximum(ax-upper,0))),
                 float(np.max(np.maximum(lb-x,0))),float(np.max(np.maximum(x-ub,0))),
                 float(np.max(abs(x[integer>0]-np.rint(x[integer>0])))) if integer.any() else 0.)
    log['raw_violation']=residual
    if residual > 1e-6:
        return Solution(False,None,{},log)
    charge,discharge=np.maximum(x[c],0),np.maximum(x[s],0)
    waste=np.maximum(x[w],0)
    delta=np.minimum(charge,discharge/.81)
    charge=charge-delta;discharge=discharge-.81*delta;waste=waste+.19*delta
    # z and w at an optimal node cannot both be positive with p>0. Cancel
    # any incumbent overlap without changing q, storage state or other actions.
    emergency=np.maximum(x[z],0)
    overlap=np.minimum(emergency,waste);emergency-=overlap;waste-=overlap
    orders={key:float(max(x[j],0)) for key,j in lookup.items()}
    Q=np.array([orders[keys[j]] for j in qi])
    # Restore tiny solver-bound deviations locally, never modify Q or prior E.
    E=np.empty(K)
    for i in range(K):
        before=energy if parents[i]<0 else E[parents[i]]
        charge[i]=min(charge[i],port,max(0.,(10800-before)/.9),max(0.,Q[i]-net[i]))
        discharge[i]=min(discharge[i],port,max(0.,(before-1200)*.9))
        E[i]=before+.9*charge[i]-discharge[i]/.9
    supply=Q-charge+discharge
    emergency=np.maximum(net-supply,0);waste=np.maximum(supply-net,0)
    fixed_cost=float(np.sum(prob[fixed_node]*p[fixed_node]*Q[fixed_node]))
    restored_objective=float(np.sum(prob*p*(Q+5*emergency))-fixed_cost)
    physics=max(float(np.max(abs(Q-charge+discharge+emergency-waste-net))),
                float(np.max(np.maximum(1200-E,0))),float(np.max(np.maximum(E-10800,0))))
    bad_modes=bool(np.any((charge>1e-6)&((discharge>1e-6)|(emergency>1e-6))))
    change=abs(restored_objective-float(res.fun))
    log.update(restored_objective=restored_objective,recovery_objective_change=change,
               recovery_nodes=int(np.count_nonzero(delta>1e-9)),physical_violation=physics,
               total_seconds=perf_counter()-begin)
    if physics>1e-6 or bad_modes or restored_objective>float(res.fun)+.01:
        log['rejection']='physical recovery failed or objective increased >0.01'
        return Solution(False,None,{},log)
    log['usable']=True
    return Solution(True,dict(q=Q,c=charge,s=discharge,z=emergency,w=waste,E=E),orders,log)
