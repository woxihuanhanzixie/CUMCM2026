#!/usr/bin/env python3
"""Q4-3 visualisation (5 figures) from Q4/outputs/Q43 ledger + summaries."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs' / 'Q43'
FIGS = OUT / 'figs'
FIGS.mkdir(parents=True, exist_ok=True)

C_NORMAL, C_ADJ, C_EMG = "#5b9bd5", "#f39c12", "#c0392b"
BRANCHES = [('C0', 'P1'), ('C1', 'P1'), ('C2', 'P1'), ('C2', 'P7')]
LABEL = {'C0-P1': 'C0-P1 0点预报锁单', 'C1-P1': 'C1-P1 四点预报锁单',
         'C2-P1': 'C2-P1 四点预报+调整（主）', 'C2-P7': 'C2-P7 上周同段价'}
COLOR = {'C0-P1': '#8c8c8c', 'C1-P1': '#5b9bd5', 'C2-P1': '#c0392b', 'C2-P7': '#f39c12'}
W = 1e4


def load_ledger() -> pd.DataFrame:
    df = pd.read_csv(OUT / 'ledger.csv')
    df['branch'] = df['policy'] + '-' + df['method']
    df['date'] = pd.Timestamp('2025-01-01') + pd.to_timedelta(df['position'] * 10, unit='m')
    return df


def fig1_annual_stacked(df: pd.DataFrame) -> Path:
    feb = df[df['position'] >= 31 * 144]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    order = ['C0-P1', 'C1-P1', 'C2-P1', 'C2-P7']
    normal = [feb[feb['branch'] == b]['normal'].sum() for b in order]
    adj = [feb[feb['branch'] == b]['adjustment'].sum() for b in order]
    emg = [feb[feb['branch'] == b]['emergency'].sum() for b in order]
    x = np.arange(4)
    ax.bar(x, normal, 0.55, label='正常购电费', color=C_NORMAL)
    ax.bar(x, adj, 0.55, bottom=normal, label='调整附加费', color=C_ADJ)
    ax.bar(x, emg, 0.55, bottom=np.array(normal) + np.array(adj), label='紧急购电费', color=C_EMG)
    for i, (n, a_, e) in enumerate(zip(normal, adj, emg)):
        tot = n + a_ + e
        ax.text(i, tot + 6e4, f'{tot/1e4:.0f}万', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[b] for b in order], fontsize=8)
    ax.set_ylabel('费用（元）')
    ax.set_title('Q4-3 四分支年度购电费用构成（2–12月，波动电价）')
    ax.legend(ncol=3, fontsize=9, frameon=False, loc='upper right')
    ax.yaxis.set_major_formatter(lambda v, _: f'{v/1e4:.0f}万')
    ax.grid(axis='y', alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    out = FIGS / 'fig_q43_annual_stacked.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig2_monthly(df: pd.DataFrame, monthly: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for b in ['C0-P1', 'C1-P1', 'C2-P1', 'C2-P7']:
        m = monthly[monthly['branch'] == b].sort_values('month')
        ax.plot(m['month'], m['total'] / 1e4, marker='o', ms=4, label=LABEL[b],
                color=COLOR[b], lw=2 if b == 'C2-P1' else 1.4)
    ax.set_xlabel('月份')
    ax.set_ylabel('月总费用（万元）')
    ax.set_title('Q4-3 四分支逐月总费用（波动电价）')
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    out = FIGS / 'fig_q43_monthly.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig3_delta(df: pd.DataFrame) -> Path:
    feb = df[df['position'] >= 31 * 144]
    tot = {b: feb[feb['branch'] == b]['total'].sum() for b in ['C0-P1', 'C1-P1', 'C2-P1', 'C2-P7']}
    deltas = [('Δ_info\nC0−C1', tot['C0-P1'] - tot['C1-P1']),
              ('Δ_adjust\nC1−C2', tot['C1-P1'] - tot['C2-P1']),
              ('Δ_price\nC2-P1−C2-P7', tot['C2-P1'] - tot['C2-P7'])]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    x = np.arange(3)
    vals = [d[1] for d in deltas]
    colors = ['#5b9bd5' if v > 0 else '#c0392b' for v in vals]
    ax.bar(x, vals, 0.5, color=colors)
    ax.axhline(0, color='k', lw=0.8)
    for i, (label, v) in enumerate(deltas):
        ax.text(i, v + (3e4 if v > 0 else -3e4), f'{v:+,.0f}', ha='center',
                va='bottom' if v > 0 else 'top', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in deltas], fontsize=9)
    ax.set_ylabel('费用差（元）')
    ax.set_title('Q4-3 对照差额（正=后者更省）')
    ax.grid(axis='y', alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    out = FIGS / 'fig_q43_delta.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig4_specified_days(df: pd.DataFrame) -> Path:
    days = {78: '2025-03-20', 171: '2025-06-21', 265: '2025-09-23', 354: '2025-12-21'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, (day, dstr) in zip(axes.flat, days.items()):
        m = df[(df['branch'] == 'C2-P1') & (df['position'] >= day * 144)
               & (df['position'] < (day + 1) * 144)]
        hour = np.arange(144) / 6
        q0 = m['q0'].values
        a = m['a_final'].values
        g = m['g'].values
        price = m['price'].values
        ax.plot(hour, q0, ls='--', color='#8c8c8c', lw=1.3, label='0点计划 q0')
        ax.plot(hour, a, color='#c0392b', lw=1.5, label='最终正常购电 a')
        diff = np.abs(a - q0) > 1e-6
        ax.scatter(hour[diff], a[diff], s=13, color='#f39c12', zorder=5, label='调整段',
                   edgecolors='k', linewidths=0.3)
        ax2 = ax.twinx()
        ax2.plot(hour, price, color='#2e86c1', lw=0.8, alpha=0.55, label='实际电价(右轴)')
        ax2.set_ylabel('电价（元/kWh）', color='#2e86c1', fontsize=8)
        ax2.tick_params(axis='y', labelsize=8, colors='#2e86c1')
        ax2.set_ylim(0, price.max() * 1.2)
        if g.max() > 1e-6:
            ax.bar(hour, g, width=1 / 6, color='#7b241c', alpha=0.35, label='紧急购电 g')
        ax.set_xticks(range(0, 25, 4))
        ax.set_xticklabels([f'{h}:00' for h in range(0, 25, 4)], fontsize=8)
        ax.set_xlim(0, 24)
        ax.set_ylim(0, max(q0.max(), a.max()) * 1.2)
        ax.set_xlabel('时刻')
        ax.set_ylabel('购电量（kWh/10min）', fontsize=9)
        tot = m['total'].sum()
        ax.set_title(f'{dstr}  （总费 {tot:,.0f} 元）', fontsize=10)
        ax.grid(alpha=0.2)
        if ax is axes.flat[0]:
            h, l = ax.get_legend_handles_labels()
            ax.legend(h, l, fontsize=8, loc='upper left', frameon=False)
    fig.suptitle('C2-P1 主方案四指定日购电计划与调整（波动电价，10分钟粒度）', fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIGS / 'fig_q43_specified_days.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig5_soc(df: pd.DataFrame) -> Path:
    c2 = df[df['branch'] == 'C2-P1'].sort_values('position')
    eod = c2.groupby(c2['position'] // 144)['E_after'].last()
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(eod.index, eod.values, color='#c0392b', lw=1.5, label='C2-P1 日末 SOC')
    ax.axhline(1200, color='#c0392b', ls=':', lw=1, alpha=0.7)
    ax.axhline(10800, color='#2e86c1', ls=':', lw=1, alpha=0.7)
    ax.text(3, 1250, '下限 1200', fontsize=8, color='#c0392b')
    ax.text(3, 10900, '上限 10800', fontsize=8, color='#2e86c1')
    feb1 = eod.iloc[31]
    ax.scatter([31], [feb1], color='#c0392b', s=26, zorder=5)
    ax.annotate(f'Feb1 SOC = {feb1:.1f}', xy=(31, feb1), xytext=(40, 6200),
                fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.8))
    ax.annotate(f'年末 SOC = {eod.iloc[-1]:.1f}', xy=(364, eod.iloc[-1]),
                xytext=(250, 2200), fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.8))
    ax.set_xlabel('日序号（0 = 1月1日）')
    ax.set_ylabel('日末储电量 SOC（kWh）')
    ax.set_title('C2-P1 全年日末 SOC 轨迹（波动电价）')
    ax.set_xlim(0, 364)
    ax.set_ylim(0, 11800)
    ticks = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 364]
    lbl = ['1/1', '2/1', '3/1', '4/1', '5/1', '6/1', '7/1', '8/1', '9/1', '10/1', '11/1', '12/1', '12/31']
    ax.set_xticks(ticks)
    ax.set_xticklabels(lbl, fontsize=8)
    ax.legend(fontsize=9, frameon=False, loc='lower right')
    ax.grid(alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    out = FIGS / 'fig_q43_soc.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    df = load_ledger()
    monthly = pd.read_csv(OUT / 'monthly_summary.csv')
    monthly['branch'] = monthly['policy'] + '-' + monthly['method']
    outs = [fig1_annual_stacked(df), fig2_monthly(df, monthly), fig3_delta(df),
            fig4_specified_days(df), fig5_soc(df)]
    print('generated:')
    for o in outs:
        print(f'  {o.name}  ({o.stat().st_size/1024:.0f} KB)')


if __name__ == '__main__':
    main()
