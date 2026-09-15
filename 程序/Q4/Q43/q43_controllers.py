#!/usr/bin/env python3
"""Q4-3 controller: E1-LP10 four-branch control with P1/P7 price forecasts.

Adapted from teammate Q4 P1 prereq_tests/controllers.py Q3 class. Changes:
  * start/end extended to the full annual domain (0..52560); the two-day
    fixture restriction removed (annual only).
  * Q42 / M0 / R2 / greedy imports removed (Q4-3 branches only).
"""
from __future__ import annotations

import copy
import time
from datetime import datetime, timedelta

import numpy as np

from q43_runtime import View, Prices, Bases, digest
from core import Problem, solve_lp, execute, LO
from controller import Controller
from forecasting import fused_curve


def normalize_orders(q, phat, events, r):
    q = np.asarray(q, float).copy()
    bad = q < 0
    if np.any(q < -1e-9):
        raise ValueError('negative order exceeds original numerical-zero policy')
    if bad.any():
        bound = float(1.5 * np.max(phat) * (-q[bad]).sum())
        if bound > 1e-6:
            raise ValueError('predicted normalization fee bound exceeded')
        events.append(dict(r=r, kind='order_zero', indices=np.flatnonzero(bad),
                           raw=q[bad].copy(), forecast_fee_bound=bound))
        q[bad] = 0
    return q


class Q3(Controller):
    """Annual E1-LP10 controller for one branch (C0/C1/C2) and one price method."""

    def __init__(self, view: View, budget, identity, bases: Bases, prices: Prices,
                 method: str, start: int = 0):
        super().__init__(view, budget, identity)
        if start != 0:
            raise ValueError('annual Q4-3 only supports start=0')
        self.bases = bases
        self.prices = prices
        self.method = method
        self.start = start
        self.end = 52560
        self.last_price = None
        self.corrections = []

    def forecast(self, day, slot):
        if day == 0:
            return super().forecast(day, slot)
        if slot == 0:
            self.load_base, self.pv_base = self.bases.get(self.access, day)
            self.base_id = digest(self.bases.refs[(self.access.mapping, day)])
            self.history_cutoff = day * 144
        if self.access.policy != 'C0' or slot == 0:
            official = self.access.official_released_by(day, slot)
            anchor = float(self.access.observations_completed_by(self.position, 'pv')[-1])
            self.pv_curve = fused_curve(self.pv_base, slot, official, anchor)
            self.official_version = (day, slot)
        self.forecasts.append(dict(position=self.position,
                                   official_version=self.official_version,
                                   pv=self.pv_curve.copy(), load=self.load_base.copy(),
                                   history_cutoff=self.history_cutoff, base_id=self.base_id))

    def problem(self, day, slot, stage):
        count = min((day + 2) * 144, 52560) - self.position
        ell = self.load_base[slot:slot + count] / 6
        pv = self.pv_curve[slot:slot + count] / 6
        phat, meta = self.prices.predict(
            self.access, np.arange(self.position, self.position + count),
            (ell - pv) * 6 / 1000, self.method,
            dict(base_id=self.base_id, official_version=self.official_version))
        self.last_price = meta
        return Problem(ell, pv, phat, self.energy, 144 - slot, stage,
                       None if stage == 'original' else self.q0[slot:],
                       None if stage == 'original' else self.a[slot:])

    def publish(self, day, slot, orders, kind):
        prices = np.ones(len(orders)) if self.last_price is None else self.last_price['prices'][:len(orders)]
        q = normalize_orders(orders, prices, self.corrections, self.position)
        return super().publish(day, slot, q, kind)

    def step(self):
        i = self.position
        day, slot = divmod(i, 144)
        if not self.start <= i < self.end or self.access.position != i:
            raise ValueError('outside annual domain')
        tick = time.perf_counter()
        if slot in (0, 36, 72, 108):
            self.forecast(day, slot)
        self.budget.addtime('prediction', time.perf_counter() - tick)
        if i == 0:
            self.publish(0, 0, np.zeros(144), 'original')
        if slot % 6 == 0 and i >= 36:
            stage = 'original' if slot == 0 else ('adjust' if slot in (36, 72, 108)
                                                  and self.access.policy == 'C2' else 'feedback')
            p = self.problem(day, slot, stage)
            result = solve_lp(p, self.budget, f'{self.access.policy}_{self.method}_{i}_{stage}')
            if stage in ('original', 'adjust'):
                self.publish(day, slot, result['A'][:144 - slot], stage)
            self.references = result['E'][:6].copy()
            self.plan_time = i
            self.plans.append(dict(position=i, stage=stage, problem=p.record(),
                                   solution=result, price_record=self.last_price,
                                   official_version=self.official_version,
                                   history_cutoff=self.history_cutoff, base_id=self.base_id))
        tick = time.perf_counter()
        a = float(self.a[slot])
        q = float(self.q0[slot])
        e = self.energy
        self.access.commit(i, a)
        load, pv, price = self.access.delivered()
        R = float(self.references[slot % 6]) if i >= 36 else LO
        act = execute(e, a, load / 6, pv / 6, R)
        self.energy = act['E']
        version = self.day_versions[-1]
        start = datetime(2025, 1, 1) + timedelta(minutes=10 * i)
        row = dict(position=i, policy=self.access.policy, method=self.method,
                   mapping='Q43', q0=q, a_final=a, ell=load / 6, pv=pv / 6,
                   price=price, E_before=e, E_after=self.energy, R=R,
                   plan_time=self.plan_time, version_id=version['id'],
                   published=version['published_position'],
                   **{k: act[k] for k in ('c', 's', 'g', 'w')})
        row.update(normal=price * a, adjustment=.5 * price * abs(a - q),
                   emergency=5 * price * act['g'])
        row['total'] = row['normal'] + row['adjustment'] + row['emergency']
        self.ledger.append(row)
        self.access.complete()
        self.position += 1
        self.budget.addtime('execution', time.perf_counter() - tick)
        self.budget.check()
        return row

    def checkpoint(self):
        return dict(base=super().snapshot(), method=self.method, start=self.start,
                    end=self.end, last_price=self.last_price, corrections=self.corrections)

    def load_checkpoint(self, obj):
        if obj['method'] != self.method or obj['start'] != self.start or obj['end'] != self.end:
            raise ValueError('checkpoint policy/domain mismatch')
        super().restore(obj['base'])
        self.last_price = copy.deepcopy(obj['last_price'])
        self.corrections = copy.deepcopy(obj['corrections'])


def make_branch(branch, data, budget, identity, bases, prices):
    name, method = branch
    view = View(data, name, 0)
    return Q3(view, budget, identity, bases, prices, method)
