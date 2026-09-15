#!/usr/bin/env python3
"""Q2 v2 causal ridge forecast: direct horizons k=0/1/2, strictly no future leak.

Frozen feature set (per series, per (h, t) row):
  [Y[h-1,t], Y[h+k-7,t], Y[h+k-14,t], 3-day mean, 7-day mean,
   3-day minus prior-3-day mean, prior-day mean,
   sin/cos(2*pi*j*t/144) j=1..3, weekday one-hot (load only, Monday reference)]
Weights w = 2^(-((d-1)-h-k)/14); weighted standardization; ridge lambda = 1.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .data_io import DT, T

LAM = 1.0
WINDOW_TARGET_DAYS = 28
MIN_TRAIN_DAYS = 7
REQUIRED_HISTORY_DAYS = 14
WEIGHT_HALF_LIFE_DAYS = 14
HARMONICS = (1, 2, 3)


def training_window(d: int, k: int) -> tuple[int, int]:
    """Publish day d (0-based), horizon k: training days h in [lo, hi)."""
    lo = max(REQUIRED_HISTORY_DAYS, d - WINDOW_TARGET_DAYS - k)
    hi = max(lo, d - k)
    return lo, hi


def day_weight(d: int, h: int, k: int) -> float:
    """Weight of training day h: 1 at h = d-1-k, halving every 14 days back."""
    return 2.0 ** (-((d - 1) - h - k) / WEIGHT_HALF_LIFE_DAYS)


def _day_mean(y: np.ndarray, a: int, b: int, h: int) -> np.ndarray:
    a = max(a, 0)
    b = min(b, h)
    if b <= a:
        return np.zeros(y.shape[1])
    return y[a:b].mean(axis=0)


def feature_rows(y: np.ndarray, h: int, k: int, weekday: int, load: bool) -> np.ndarray:
    """Feature rows (one per slot) predicting y[h+k] from history up to day h."""
    lag1 = y[h - 1] if h >= 1 else np.zeros(y.shape[1])
    src7 = h + k - 7
    src14 = h + k - 14
    lag7 = y[src7] if src7 >= 0 else np.zeros(y.shape[1])
    lag14 = y[src14] if src14 >= 0 else np.zeros(y.shape[1])
    m3 = _day_mean(y, h - 3, h, h)
    m7 = _day_mean(y, h - 7, h, h)
    m3_prior = _day_mean(y, h - 6, h - 3, h)
    day_mean = float(y[h - 1].mean()) if h >= 1 else 0.0
    t = np.arange(y.shape[1], dtype=float)
    cols = [lag1, lag7, lag14, m3, m7, m3 - m3_prior, np.full(y.shape[1], day_mean)]
    for j in HARMONICS:
        cols.append(np.sin(2 * np.pi * j * t / y.shape[1]))
        cols.append(np.cos(2 * np.pi * j * t / y.shape[1]))
    if load:
        for w in range(1, 7):
            cols.append(np.full(y.shape[1], 1.0 if weekday == w else 0.0))
    return np.column_stack(cols)


def fit_series(y: np.ndarray, d: int, k: int, dates, load: bool) -> dict[str, Any] | None:
    lo, hi = training_window(d, k)
    if hi - lo < MIN_TRAIN_DAYS:
        return None
    blocks_x, blocks_y, blocks_w = [], [], []
    for h in range(lo, hi):
        blocks_x.append(feature_rows(y, h, k, dates[h + k].weekday(), load))
        blocks_y.append(y[h + k])
        blocks_w.append(np.full(y.shape[1], day_weight(d, h, k)))
    X = np.vstack(blocks_x)
    target = np.concatenate(blocks_y)
    w = np.concatenate(blocks_w)
    b = float(np.average(target, weights=w))
    mu = np.average(X, axis=0, weights=w)
    sd = np.sqrt(np.average((X - mu) ** 2, axis=0, weights=w))
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = (X - mu) / sd
    WZ = Z * w[:, None]
    beta = np.linalg.solve(Z.T @ WZ + LAM * np.eye(Z.shape[1]), WZ.T @ (target - b))
    return {'b': b, 'mu': mu, 'sd': sd, 'beta': beta, 'n': hi - lo}


def predict_series(y: np.ndarray, d: int, k: int, dates, load: bool, fit: dict[str, Any]) -> np.ndarray:
    X = feature_rows(y, d, k, dates[d + k].weekday(), load)
    Z = (X - fit['mu']) / fit['sd']
    return np.clip(fit['b'] + Z @ fit['beta'], 0.0, None)


def fallback_load(y: np.ndarray, d: int, k: int) -> np.ndarray | None:
    src = d + k - 7
    if src >= 0:
        return y[src].astype(float).copy()
    src = d - 1
    if src >= 0:
        return y[src].astype(float).copy()
    return None


def fallback_pv(y: np.ndarray, d: int, k: int) -> np.ndarray | None:
    src = d - 1
    if src >= 0:
        return y[src].astype(float).copy()
    return None


class RidgeForecaster:
    """Per-day, per-horizon ridge fit using only days strictly before publish day d."""

    def __init__(self, data: dict[str, Any], horizons: tuple[int, ...] = (0, 1, 2)):
        self.load_act = data['load_act']
        self.pv_act = data['pv_act']
        self.dates = data['dates']
        self.horizons = tuple(horizons)
        self.load_a1 = data.get('load_a1')
        self.pv_a1 = data.get('pv_a1')

    def n_train(self, d: int, k: int) -> int:
        lo, hi = training_window(d, k)
        return hi - lo

    def _predict_series(self, y: np.ndarray, d: int, k: int, load: bool,
                        a1: np.ndarray | None) -> np.ndarray:
        if d + k >= len(y):
            raise ValueError(f'target day {d + k} out of range for horizon {k}')
        fit = fit_series(y, d, k, self.dates, load)
        if fit is None:
            fb = fallback_load(y, d, k) if load else fallback_pv(y, d, k)
            if fb is not None:
                return fb
            if a1 is not None:
                return a1.astype(float).copy()
            raise ValueError(f'fallback S has no history at publish day {d}, horizon {k}')
        return predict_series(y, d, k, self.dates, load, fit)

    def predict_day(self, d: int, k: int) -> dict[str, np.ndarray]:
        return {
            'load': self._predict_series(self.load_act, d, k, True, self.load_a1),
            'pv': self._predict_series(self.pv_act, d, k, False, self.pv_a1),
        }

    def forecast_net_kwh(self, d: int, k: int) -> np.ndarray:
        pr = self.predict_day(d, k)
        return (pr['load'] - pr['pv']) * DT
