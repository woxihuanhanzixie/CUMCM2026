#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""问题一结果可视化。

只读取 solve/evaluate 阶段已经生成的结果文件，不重新求解、不修改模型。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
DT_H = 1.0 / 6.0
ETA_C = 0.9
ETA_D = 0.9
SOC_LB = 1200.0
SOC_UB = 10800.0

C_LOAD = "#1f3b6f"
C_PV = "#e8a33d"
C_NET = "#2e8b57"
C_PRICE = "#7b2d5e"
C_BUY = "#4472c4"
C_CHARGE = "#c0392b"
C_DISCHARGE = "#16a085"
C_SOC = "#2f5fa8"

DPI = 300


def save(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def configure_font() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    preferred = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans CN"]
    chosen = [name for name in preferred if name in available]
    if not chosen:
        raise RuntimeError("未找到中文字体，图中的中文会显示为方框")
    plt.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["figure.dpi"] = 100
    return chosen[0]


def load_results():
    trace = pd.read_csv(HERE / "q1_trace.csv")
    evalj = json.loads((HERE / "q1_evaluation.json").read_text(encoding="utf-8"))
    eta = pd.read_csv(HERE / "q1_sensitivity_eta.csv")
    soc = pd.read_csv(HERE / "q1_sensitivity_soc.csv")
    return trace, evalj, eta, soc


def self_check(trace: pd.DataFrame, metrics: dict) -> tuple[list[str], dict]:
    problems = []
    if len(trace) != 144:
        problems.append(f"trace 行数为 {len(trace)}，应为 144")

    balance = trace["q"] + trace["pv_energy"] + trace["s"] - trace["load_energy"] - trace["c"] - trace["w"]
    rec = trace["E_end"] - (trace["E_start"] + ETA_C * trace["c"] - trace["s"] / ETA_D)
    rec_first = trace["E_start"].iloc[0] - metrics["soc_initial"]
    if abs(balance).max() > 1e-7:
        problems.append(f"功率平衡残差 {abs(balance).max():.3e} > 1e-7")
    if abs(rec).max() > 1e-7:
        problems.append(f"储能递推残差 {abs(rec).max():.3e} > 1e-7")
    if abs(rec_first) > 1e-7:
        problems.append(f"初始储电量不符：{rec_first:.3e}")
    if trace["E_end"].iloc[-1] - metrics["soc_terminal"] > 1e-7:
        problems.append("末端储电量不为 6000 kWh")
    if ((trace["q"] < -1e-9) | (trace["w"] < -1e-9)).any():
        problems.append("购电量或弃光量出现负值")

    objective = float(np.sum(trace["purchase_cost"]))
    if abs(objective - metrics["objective"]) > 1e-6:
        problems.append(f"逐段费用合计 {objective:.6f} 与目标值 {metrics['objective']:.6f} 不一致")

    baseline = float(np.sum(trace["price"] * np.maximum(trace["load_energy"] - trace["pv_energy"], 0.0)))
    if abs(baseline - metrics["benchmark_cost"]) > 1e-6:
        problems.append(f"无储能基准重算 {baseline:.6f} 与 evaluation 记录 {metrics['benchmark_cost']:.6f} 不一致")

    if ((trace["c"] > 1e-7) & (trace["s"] > 1e-7)).any():
        problems.append("存在同时充放电的时段")
    stats = {
        "balance_max": float(abs(balance).max()),
        "recursion_max": float(abs(rec).max()),
        "objective": objective,
        "benchmark": baseline,
    }
    return problems, stats


def style_time_axis(ax, xlabel="时刻（时）"):
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(True, ls=":", lw=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def hour_of_slot(trace):
    return trace["t"].to_numpy() * DT_H


def hour_of_center(trace):
    return (trace["t"].to_numpy() - 0.5) * DT_H


def fig1_inputs(trace, path):
    x = hour_of_center(trace)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2], "hspace": 0.09}
    )
    ax1.plot(x, trace["load_power"], drawstyle="steps-mid", color=C_LOAD, lw=1.6, label="负荷功率")
    ax1.plot(x, trace["pv_power"], drawstyle="steps-mid", color=C_PV, lw=1.6, label="光伏功率")
    ax1.fill_between(x, trace["pv_power"], step="mid", color=C_PV, alpha=0.22)
    ax1.plot(x, trace["load_power"] - trace["pv_power"], ls="--", color=C_NET, lw=1.4, label="净负荷（负荷−光伏）")
    ax1.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax1.set_ylabel("功率（kW）")
    ax1.set_title("(a) 负荷、光伏与净负荷功率曲线（144 个 10 分钟时段）")
    ax1.legend(frameon=False, loc="upper left", ncol=1)

    ax2.step(x, trace["price"], where="mid", color=C_PRICE, lw=1.6)
    ax2.fill_between(x, trace["price"], step="mid", color=C_PRICE, alpha=0.16)
    ax2.set_ylabel("电价（元/kWh）")
    ax2.set_title(f"(b) 分时购电电价：{trace['price'].min():.4f} ~ {trace['price'].max():.4f} 元/kWh")
    style_time_axis(ax2)
    fig.text(0.995, 0.005, "时间戳按区间末端标注，功率为该区间代表值", ha="right", fontsize=7.5, color="0.4")
    save(fig, path)


def fig2_supply(trace, metrics, path):
    x = hour_of_center(trace)
    buy_p = trace["q"].to_numpy() / DT_H
    pv_p = trace["pv_power"].to_numpy()
    dis_p = trace["discharge_power"].to_numpy()
    net_p = trace["load_power"].to_numpy() - pv_p
    demand_p = trace["load_power"].to_numpy() + trace["charge_power"].to_numpy() + trace["w"].to_numpy() / DT_H
    stack_top = buy_p + pv_p + dis_p

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(10, 8.6), sharex=True, gridspec_kw={"height_ratios": [1, 1], "hspace": 0.1}
    )
    ax.stackplot(
        x, buy_p, pv_p, dis_p,
        colors=[C_BUY, C_PV, C_DISCHARGE], alpha=0.75, step="mid", zorder=1,
        labels=["计划购电功率", "光伏功率", "储能放电功率"],
    )
    ax.plot(x, demand_p, drawstyle="steps-mid", color="black", lw=2.0, zorder=4,
            label="总需求（负荷+充电+弃光）")
    ax.plot(x, trace["load_power"], ls="--", color="0.25", lw=1.6, zorder=5, label="负荷功率")
    ax.set_ylabel("功率（kW）")
    ax.set_ylim(0, stack_top.max() * 1.28)
    ax.set_title("(a) 最优调度下的电源侧出力构成与供需平衡")
    ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=8.5)
    style_time_axis(ax, xlabel=None)

    ax2.fill_between(x, net_p, buy_p, step="mid", color=C_BUY, alpha=0.22, label="储能平移的功率（购电与净负荷之差）")
    ax2.plot(x, net_p, drawstyle="steps-mid", color=C_NET, lw=1.9, label="净负荷功率")
    ax2.plot(x, buy_p, drawstyle="steps-mid", color=C_BUY, lw=1.9, label="计划购电功率")
    ax2.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax2.set_ylabel("功率（kW）")
    ax2.set_xlabel("时刻（时）")
    ax2.set_title("(b) 储能削峰填谷：购电功率相对净负荷被拉平")
    ax2.legend(frameon=False, loc="lower left", ncol=1, fontsize=8.5)
    style_time_axis(ax2)
    fig.text(0.5, 0.005, "上图负荷虚线与黑色总需求线之间的间距即储能充电功率；各时段均满足 购电+光伏+放电 = 负荷+充电+弃光，"
                         f"全天弃光 {trace['w'].sum() * DT_H:.0f} kWh，光伏利用率 {metrics['pv_utilization_percent']:.0f}%"
                         "（时间戳按区间末端标注）",
             ha="center", fontsize=8, color="0.35")
    save(fig, path)


def fig3_soc(trace, metrics, path):
    xb = hour_of_slot(trace)
    soc = np.concatenate(([trace["E_start"].iloc[0]], trace["E_end"].to_numpy()))
    x_edge = np.arange(len(soc)) * DT_H
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08}
    )
    ax1.axhspan(SOC_LB, SOC_UB, color="0.88", alpha=0.45, label="允许储电量区间")
    ax1.plot(x_edge, soc, color=C_SOC, lw=2.1, zorder=5, label="储能储电量 $E_t$")
    ax1.fill_between(x_edge, soc, SOC_LB, color=C_SOC, alpha=0.16)
    ax1.axhline(metrics["soc_initial"], ls="--", color="0.35", lw=1.1)
    ax1.text(0.15, metrics["soc_initial"] + 160, "初始 / 末端 6000 kWh", fontsize=8.5, color="0.3")
    hit_max = soc >= SOC_UB - 1e-6
    hit_min = soc <= SOC_LB + 1e-6
    if hit_max.any():
        ax1.scatter(x_edge[hit_max], soc[hit_max], marker="^", s=34, color=C_DISCHARGE, zorder=6,
                    label=f"触及上限 10800（{metrics['soc_max_binding_count']} 段）")
    if hit_min.any():
        ax1.scatter(x_edge[hit_min], soc[hit_min], marker="v", s=34, color=C_CHARGE, zorder=6,
                    label=f"触及下限 1200（{metrics['soc_min_binding_count']} 段）")
    ax1.set_ylabel("储电量（kWh）")
    ax1.set_title("(a) 储能荷电量全天轨迹")
    ax1.set_ylim(0, SOC_UB * 1.13)
    ax1.legend(frameon=False, loc="lower left", ncol=2, fontsize=8.5)
    style_time_axis(ax1, xlabel=None)

    ax2.bar(xb - DT_H / 2, trace["charge_power"], width=DT_H * 0.92, color=C_CHARGE, alpha=0.85, label="充电功率")
    ax2.bar(xb - DT_H / 2, -trace["discharge_power"], width=DT_H * 0.92, color=C_DISCHARGE, alpha=0.85,
            label="放电功率（取负显示）")
    ax2.axhline(5000, ls=":", color=C_CHARGE, lw=1.1)
    ax2.axhline(-5000, ls=":", color=C_DISCHARGE, lw=1.1)
    ax2.text(23.9, 5250, "功率上限 5000 kW", ha="right", fontsize=8, color=C_CHARGE)
    ax2.set_ylabel("充/放电功率（kW）")
    ax2.set_title(f"(b) 充电 {trace['c'].sum():.0f} kWh ／ 放电 {trace['s'].sum():.0f} kWh"
                  f"（损耗 {metrics['charge_discharge_loss']:.0f} kWh）")
    style_time_axis(ax2)
    ax2.set_ylim(-6600, 9200)
    ax2.legend(frameon=False, loc="upper left", ncol=2)
    save(fig, path)


def fig4_cost(trace, metrics, path):
    x = hour_of_center(trace)
    base_cost = trace["price"].to_numpy() * np.maximum(trace["load_energy"] - trace["pv_energy"], 0.0)
    opt_cost = trace["purchase_cost"].to_numpy()
    saving = np.maximum(base_cost - opt_cost, 0.0)
    extra = np.maximum(opt_cost - base_cost, 0.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5), gridspec_kw={"width_ratios": [3, 2], "wspace": 0.22})
    ax1.fill_between(x, opt_cost, opt_cost + saving, step="mid", color="#bcd4ee",
                     label=f"储能省费时段（合计 {saving.sum():,.0f} 元）")
    ax1.fill_between(x, base_cost, base_cost + extra, step="mid", color="#f2c9c4",
                     label=f"储能增支时段（合计 {extra.sum():,.0f} 元）")
    ax1.plot(x, base_cost, drawstyle="steps-mid", ls="--", color="0.45", lw=1.5, label="无储能基准费用")
    ax1.plot(x, opt_cost, drawstyle="steps-mid", color=C_BUY, lw=1.8, label="问题一最优费用")
    ax1.set_xlabel("时刻（时）")
    ax1.set_ylabel("10 分钟时段购电费用（元）")
    ax1.set_title("(a) 逐时段购电费用对比")
    style_time_axis(ax1)
    ax1.legend(frameon=False, loc="upper left")

    bars = ax2.bar([0, 1], [metrics["benchmark_cost"], metrics["objective"]],
                   width=0.5, color=["0.62", C_BUY])
    for rect, value in zip(bars, [metrics["benchmark_cost"], metrics["objective"]]):
        ax2.text(rect.get_x() + rect.get_width() / 2, value + 600, f"{value:,.0f} 元", ha="center", fontsize=9.5)
    ax2.annotate(
        "",
        xy=(1.38, metrics["objective"]),
        xytext=(1.38, metrics["benchmark_cost"]),
        arrowprops=dict(arrowstyle="<->", color=C_CHARGE, lw=1.4),
    )
    ax2.text(1.44, (metrics["objective"] + metrics["benchmark_cost"]) / 2,
             f"↓ 节省 {metrics['saving_cost']:,.0f} 元\n   （{metrics['saving_percent']:.2f}%）",
             color=C_CHARGE, fontsize=10, va="center")
    ax2.set_xlim(-0.5, 2.15)
    ax2.set_xticks([0, 1], ["无储能基准", "含储能最优"])
    ax2.set_ylim(0, metrics["benchmark_cost"] * 1.18)
    ax2.set_ylabel("全天购电费用（元）")
    ax2.set_title("(b) 全天总费用对比")
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.grid(True, axis="y", ls=":", lw=0.7, alpha=0.65)
    ax2.set_axisbelow(True)
    save(fig, path)


def fig5_sensitivity(eta, soc, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"wspace": 0.42})

    single = eta[eta["scenario"].str.startswith("single")].sort_values("eta_charge")
    ax1.plot(single["eta_charge"] * 100, single["objective"] / 1e4, "o-", color=C_BUY, lw=2,
             label="最优费用（万元）")
    for xv, yv in zip(single["eta_charge"] * 100, single["objective"] / 1e4):
        ax1.annotate(f"{yv:.2f}", (xv, yv), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5)
    ax1b = ax1.twinx()
    ax1b.plot(single["eta_charge"] * 100, single["total_purchase"] / 1e3, "s--", color=C_DISCHARGE, lw=1.6,
              label="全天购电量（MWh）")
    for _, row in eta[~eta["scenario"].str.startswith("single")].iterrows():
        ax1b.scatter([row["eta_charge"] * 100], [row["total_purchase"] / 1e3], marker="D", s=55,
                     color=C_PRICE, zorder=6, label="往返效率 0.90 情景")
        ax1b.annotate(f"往返效率 0.90：{row['objective'] / 1e4:.2f} 万元",
                      (row["eta_charge"] * 100, row["total_purchase"] / 1e3),
                      textcoords="offset points", xytext=(10, -16), fontsize=8, color=C_PRICE)
    ax1.axvline(90, ls=":", color="0.5", lw=1)
    ax1.text(90.4, ax1.get_ylim()[0], "基准 η=0.90", fontsize=8.5, color="0.4", va="bottom")
    ax1.set_xlabel("单程效率（%，往返效率为其平方）")
    ax1.set_ylabel("最优费用（万元）", color=C_BUY)
    ax1.tick_params(axis="y", labelcolor=C_BUY)
    ax1b.set_ylabel("全天购电量（MWh）", color=C_DISCHARGE)
    ax1b.tick_params(axis="y", labelcolor=C_DISCHARGE)
    ax1.set_title("(a) 充放电效率敏感性")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="center left")
    ax1.grid(True, ls=":", lw=0.7, alpha=0.6)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1b.spines["top"].set_visible(False)

    soc = soc.sort_values("initial_and_terminal_soc")
    ax2.plot(soc["initial_and_terminal_soc"], soc["objective"], "o-", color=C_PRICE, lw=2)
    for xv, yv in zip(soc["initial_and_terminal_soc"], soc["objective"]):
        ax2.annotate(f"{yv:,.2f}", (xv, yv), textcoords="offset points", xytext=(0, 11), ha="center", fontsize=8.5)
    base = soc.loc[soc["initial_and_terminal_soc"] == 6000, "objective"].iloc[0]
    ax2.scatter([6000], [base], s=110, facecolor="none", edgecolor="black", lw=1.4, zorder=5)
    ax2.set_xlabel("初始 = 末端储电量（kWh）")
    ax2.set_ylabel("最优费用（元）")
    ax2.set_title("(b) 首末储电量约束敏感性")
    ax2.set_ylim(soc["objective"].min() - 4, soc["objective"].max() + 4)
    ax2.text(0.03, 0.06, f"三种情景全天购电量恒为 {soc['total_purchase'].iloc[0]:,.2f} kWh，\n"
                         f"仅费用相差 {soc['objective'].max() - soc['objective'].min():.2f} 元",
             transform=ax2.transAxes, fontsize=8.5, color="0.3", va="bottom")
    ax2.grid(True, ls=":", lw=0.7, alpha=0.6)
    ax2.set_axisbelow(True)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    save(fig, path)


def fig6_blocks(metrics, path):
    blocks = metrics["four_hour_blocks"]
    labels = [f"{4 * i:02d}:00–{4 * (i + 1):02d}:00" for i in range(6)]
    keys = sorted(blocks.keys())
    charge = [blocks[k]["charge"] for k in keys]
    discharge = [blocks[k]["discharge"] for k in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.bar(x - 0.2, charge, width=0.4, color=C_CHARGE, alpha=0.9, label="充电量")
    ax.bar(x + 0.2, discharge, width=0.4, color=C_DISCHARGE, alpha=0.9, label="放电量")
    for xi, (cv, dv) in enumerate(zip(charge, discharge)):
        ax.text(xi - 0.2, cv + 90, f"{cv:,.0f}", ha="center", fontsize=8)
        ax.text(xi + 0.2, dv + 90, f"{dv:,.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_xlabel("四小时区段（以区段起点标注）")
    ax.set_ylabel("电量（kWh）")
    ax.set_title("储能充电量与放电量的四小时汇总（对应 result1.xlsx“充放电量”表）")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", ls=":", lw=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_ylim(0, max(max(charge), max(discharge)) * 1.16)
    save(fig, path)


def fig7_arbitrage(trace, path):
    price = trace["price"].to_numpy()
    chg = trace["charge_power"].to_numpy()
    dis = trace["discharge_power"].to_numpy()
    buy = trace["q"].to_numpy() / DT_H
    is_chg = chg > 1e-6
    grid_chg = is_chg & (buy > 1e-6)
    pv_chg = is_chg & (buy <= 1e-6)
    is_dis = dis > 1e-6

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.scatter(price[grid_chg], chg[grid_chg], s=46, color=C_CHARGE, alpha=0.9, edgecolor="none",
               label=f"低价购电充电（{grid_chg.sum()} 段）")
    ax.scatter(price[pv_chg], chg[pv_chg], s=46, color="#f0a04b", alpha=0.9, marker="^", edgecolor="none",
               label=f"光伏盈余充电（{pv_chg.sum()} 段）")
    ax.scatter(price[is_dis], -dis[is_dis], s=46, color=C_DISCHARGE, alpha=0.9, edgecolor="none",
               label=f"储能放电（{is_dis.sum()} 段）")
    ax.axhline(0, color="black", lw=1)

    avg_chg = float(np.average(price[grid_chg], weights=chg[grid_chg]))
    avg_dis = float(np.average(price[is_dis], weights=dis[is_dis]))
    ax.axvline(avg_chg, ls="--", color=C_CHARGE, lw=1.3)
    ax.axvline(avg_dis, ls="--", color=C_DISCHARGE, lw=1.3)
    ax.text(avg_chg - 0.012, ax.get_ylim()[1] * 0.55, f"低价充电\n加权均价\n{avg_chg:.4f}",
            color=C_CHARGE, fontsize=8.5, ha="right", va="center")
    ax.text(avg_dis - 0.014, -abs(ax.get_ylim()[0]) * 0.55, f"放电\n加权均价\n{avg_dis:.4f}",
            color=C_DISCHARGE, fontsize=8.5, ha="right", va="center")

    ax.text(0.985, 0.60,
            f"度电套利利差  {avg_dis - avg_chg:.4f} 元/kWh\n"
            f"充电电量 {trace['c'].sum():,.0f} kWh，其中光伏盈余 "
            f"{trace.loc[pv_chg, 'c'].sum():,.0f} kWh\n"
            f"放电电量 {trace['s'].sum():,.0f} kWh",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.8,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.75", alpha=0.94))

    ax.set_xlabel("该时段购电电价（元/kWh）")
    ax.set_ylabel("储能功率（kW，充电为正、放电为负）")
    ax.set_title("储能充放电对电价的响应：低价购电与光伏盈余是两条独立充电通道")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.grid(True, ls=":", lw=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(0.5, 0.005, "光伏盈余时段即使电价偏高仍选择充电，因为多余光伏若不储存只能弃光；"
                         "放电则集中在傍晚 1.25~1.40 元/kWh 的尖峰时段",
             ha="center", fontsize=8, color="0.35")
    save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成问题一结果的可视化图")
    parser.add_argument("--outdir", default=str(HERE / "figures"))
    parser.add_argument("--dpi", type=int, default=300)
    global DPI
    args = parser.parse_args()
    DPI = args.dpi

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    font = configure_font()

    trace, evalj, eta, soc = load_results()
    metrics = evalj["metrics"]
    problems, stats = self_check(trace, metrics)
    if problems:
        print("结果文件自检未通过：")
        for p in problems:
            print("  -", p)
        return 1

    figures = [
        ("fig1_inputs.png", lambda p: fig1_inputs(trace, p)),
        ("fig2_supply.png", lambda p: fig2_supply(trace, metrics, p)),
        ("fig3_soc.png", lambda p: fig3_soc(trace, metrics, p)),
        ("fig4_cost.png", lambda p: fig4_cost(trace, metrics, p)),
        ("fig5_sensitivity.png", lambda p: fig5_sensitivity(eta, soc, p)),
        ("fig6_blocks.png", lambda p: fig6_blocks(metrics, p)),
        ("fig7_arbitrage.png", lambda p: fig7_arbitrage(trace, p)),
    ]
    print(f"中文字体：{font}")
    print(f"自检通过：最大平衡残差 {stats['balance_max']:.2e} kWh，最大递推残差 {stats['recursion_max']:.2e} kWh")
    print(f"  重算最优费用 {stats['objective']:.6f} 元；重算无储能基准 {stats['benchmark']:.6f} 元（均与 evaluation 一致）")
    for name, fn in figures:
        path = outdir / name
        fn(path)
        print(f"  生成 {path}  ({path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
