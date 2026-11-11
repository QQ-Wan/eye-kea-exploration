"""Figures 3, 4b, 5 and 6 of the manuscript and Supplementary Figure S1, drawn from the shared data in data/ only.

Figure 4a (example trajectories) needs raw head positions and is not redrawn here.
Group comparisons use the same tests as the text: Student's t for spatial coverage, object coverage and
search success, Welch's t for backtracking duration. Age lines are fitted after the outlier rule of each age
regression (results/participant_stats.json); the Figure S1 lines are fitted to each outcome's common sample
(common.common_sample). Excluded participants are drawn as open circles. Error bars are standard errors of the mean, and shaded bands are
95% confidence intervals of the fitted line.
Style: plot_style.py (half-open axes, font sizes keyed to the panel height), with the group colors of the submitted figures.
Run from the repository root or from analysis/. Outputs: figures/*.png and *.pdf, results/figure5_means.csv
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import (PALETTE, align_panel_labels, add_title, apply_xinger, place_legend, place_stats,  # noqa: E402
                        save_xinger, scale_for_panel, xinger_axes)
from common import common_sample  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
P = pd.read_csv(ROOT / "data/participants.csv")
S = pd.read_csv(ROOT / "data/subject_measures.csv")
E = pd.read_csv(ROOT / "data/events.csv")
PS = json.load(open(ROOT / "results/participant_stats.json"))
MAX_ENC = 4

# adaptation: per participant, slope of the change in stop duration on encounter number (MS events, encounters 1-4)
ms = E[(E.event_type == "MS") & (E.encounter <= MAX_ENC)]
adapt = {c: np.polyfit(g.encounter, g.stop_diff, 1)[0] for c, g in ms.groupby("code") if g.encounter.nunique() >= 2}
D = S.merge(P[["code", "age"]], on="code")
D["adaptation"] = D.code.map(adapt)
D["opt1"] = D.seen_cells / D.avail_cells
C = D[D.group == "child"]

GROUPS = ["child", "adult"]
NAME = {"child": "Children", "adult": "Adults"}
# group colors carried over from the submitted figures; everything else follows plot_style
COL = {"child": "#00A08A", "adult": "#F98400"}        # fills, bands, adult-mean line
COLD = {"child": "#006B5E", "adult": "#CC6600"}       # points
COLL = {"child": "#004D40", "adult": "#CC6600"}       # fitted lines
GREY = "#7F7F7F"                                      # control events
PANEL_H = 3.45                       # every panel is the reference height, so all figures share one text-to-plot ratio
SCALE = scale_for_panel(PANEL_H)
apply_xinger(scale=SCALE)
PT = (7 * SCALE) ** 2                # marker area
RNG = np.random.default_rng(1)

def save(fig, name):
    """PDF with a transparent background (style default); PNG on opaque white for the manuscript."""
    save_xinger(fig, str(OUT / name), exts=("pdf",))
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- number formatting (as in the manuscript)
def neg(s): return s.replace("-", "−")
def nz(x, d): s = f"{x:.{d}f}"; return neg(s.replace("0.", ".", 1) if s.startswith(("0.", "-0.")) else s)
def fp(p): return "p < .001" if p < .001 else f"p = {nz(p, 3)}"
def fdf(df): return f"{int(round(df))}" if abs(df - round(df)) < 1e-6 else f"{df:.1f}"
def fb(b, ci=()):
    """Decimals chosen from the estimate and its CI, as in the manuscript."""
    m = max([abs(b)] + [abs(c) for c in ci])
    d = 2 if m >= 1 else 3 if m >= .01 else 4 if m >= .001 else 5
    m = abs(b)
    while b != 0 and round(m, d) == 0: d += 1
    return neg(f"{b:.{d}f}")
def ttest_text(v, welch=False, children_first=False):
    a, c = D[D.group == "adult"][v].dropna(), D[D.group == "child"][v].dropna()
    t, p = st.ttest_ind(a, c, equal_var=not welch)
    if welch:
        va, vc = a.var() / len(a), c.var() / len(c)
        df = (va + vc) ** 2 / (va ** 2 / (len(a) - 1) + vc ** 2 / (len(c) - 1))
    else:
        df = len(a) + len(c) - 2
    if children_first: t = -t        # sign follows the order of the groups in the text
    return f"t({fdf(df)}) = {neg(f'{t:.2f}')}\n{fp(p)}"
def beta_text(r): return f"β = {fb(r['b'], r['CI'])}, t({fdf(r['df'])}) = {neg(format(r['t'], '.2f'))}\n{fp(r['p'])}"


# ---------------------------------------------------------------- panel types
def ols_band(ax, x, y, color, xlim, line=None):
    """Least-squares line with its 95% confidence band over xlim."""
    m = sm.OLS(np.asarray(y), sm.add_constant(np.asarray(x))).fit()
    xs = np.linspace(*xlim, 100)
    pr = m.get_prediction(sm.add_constant(xs)).summary_frame(alpha=0.05)
    ax.fill_between(xs, pr.mean_ci_lower, pr.mean_ci_upper, color=color, alpha=0.2, lw=0)
    ax.plot(xs, pr["mean"], color=line or color)

def bars(ax, v, ylab, ylim, welch=False, children_first=False):
    for i, g in enumerate(GROUPS):
        x = D[D.group == g][v].dropna()
        ax.bar(i, x.mean(), width=0.6, color=COL[g], alpha=0.45, lw=0)
        ax.scatter(i + RNG.uniform(-0.15, 0.15, len(x)), x, s=PT, color=COLD[g], alpha=0.85, lw=0, zorder=3)
        ax.errorbar(i, x.mean(), yerr=x.std() / np.sqrt(len(x)), color="#000000", capsize=0, lw=2.2 * SCALE, zorder=4)
    ax.set_xticks([0, 1], [NAME[g] for g in GROUPS]); ax.set_xlim(-0.55, 1.55); ax.set_ylim(*ylim)
    ax.set_ylabel(ylab)
    xinger_axes(ax, SCALE)
    place_stats(ax, ttest_text(v, welch, children_first), SCALE)

AGE_KEY = {"spatial_coverage": "age_spatial_coverage", "opt1": "age_opt1", "BT": "age_BT", "success_proportion": "age_success"}
def age_scatter(ax, v, ylim):
    r = PS[AGE_KEY[v]]; ex = C.code.isin(r["excluded"])
    ax.axhline(D[D.group == "adult"][v].mean(), color=COL["adult"], lw=2.2 * SCALE, ls="--")
    ols_band(ax, C[~ex].age, C[~ex][v], COL["child"], (6.8, 15.2), COLL["child"])
    ax.scatter(C[~ex].age, C[~ex][v], s=PT, color=COLD["child"], alpha=0.85, lw=0, zorder=3)
    ax.scatter(C[ex].age, C[ex][v], s=PT, facecolors="none", edgecolors=COLD["child"], lw=1.5 * SCALE, zorder=3)
    ax.set_xticks(range(7, 16)); ax.set_xlim(6.6, 15.4); ax.set_ylim(*ylim)
    ax.set_xlabel("Age (years)")
    xinger_axes(ax, SCALE)
    place_stats(ax, beta_text(r), SCALE)
    place_legend(ax, SCALE, italic=False, handles=[
        Line2D([], [], marker="o", ls="none", color=COLD["child"], markersize=7 * SCALE),
        Line2D([], [], color=COL["adult"], ls="--")], labels=["Children", "Adult mean"])

def top(v, step):   # axis top with room above the data for the in-panel text
    return np.ceil(D[v].max() * 1.30 / step) * step


# ---------------------------------------------------------------- Figure 3: spatial coverage and object coverage
fig = plt.figure(figsize=(10.5, 2 * PANEL_H + 1.6))
gs = GridSpec(2, 2, width_ratios=[1, 1.7], wspace=0.28, hspace=0.45, figure=fig)
a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]); b1, b2 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
bars(a1, "spatial_coverage", "Spatial coverage", (0, 1.25)); age_scatter(a2, "spatial_coverage", (0, 1.25))
bars(b1, "opt1", "Object coverage", (0, top("opt1", 0.1))); age_scatter(b2, "opt1", (0, top("opt1", 0.1)))
for ax in (a1, a2, b1, b2): ax.set_yticks([t for t in ax.get_yticks() if t <= ax.get_ylim()[1] * 0.81])
align_panel_labels([(a1, "a"), (b1, "b")], SCALE)
save(fig, "fig3_global_exploration")

# ---------------------------------------------------------------- Figure 4b: backtracking duration
fig = plt.figure(figsize=(10.5, PANEL_H + 0.9))
gs = GridSpec(1, 2, width_ratios=[1, 1.7], wspace=0.28, figure=fig)
a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
lim = (0, top("BT", 50))
bars(a1, "BT", "Backtracking duration (s)", lim, welch=True, children_first=True); age_scatter(a2, "BT", lim)
for ax in (a1, a2): ax.set_yticks([t for t in ax.get_yticks() if t <= lim[1] * 0.81])
align_panel_labels([(a1, "b")], SCALE)
save(fig, "fig4b_backtracking_duration")

# ---------------------------------------------------------------- Figure 5: stopping after MS and control events, encounters 1-4
E4 = E[E.encounter <= MAX_ENC]
s5 = (E4.groupby(["group", "event_type", "encounter"]).stop_diff
      .agg(m="mean", se=lambda x: x.std() / np.sqrt(len(x)), n="size").reset_index())
fig, axes = plt.subplots(1, 2, figsize=(10.5, PANEL_H + 0.9), sharey=True, gridspec_kw={"wspace": 0.12})
for ax, g in zip(axes, GROUPS):
    ax.axhline(0, color=PALETTE["lightgrey"], lw=1.0 * SCALE)
    for ev, col, mk, lab in [("MS", COL[g], "s", "Multisensory"), ("Control", GREY, "o", "Control")]:
        x = s5[(s5.group == g) & (s5.event_type == ev)]
        ax.errorbar(x.encounter, x.m, yerr=x.se, color=col, marker=mk, capsize=0, label=lab, markersize=8 * SCALE)
    ax.set_xticks(range(1, MAX_ENC + 1)); ax.set_xlim(0.6, MAX_ENC + 0.4)
    ax.set_xlabel("Encounter")
    if g == "child": ax.set_ylabel("Change in stop duration (s)")
    xinger_axes(ax, SCALE, wrap_labels=False)   # keep the y label on one line, as in the manuscript figure
    add_title(ax, NAME[g], SCALE)
lo, hi = axes[0].get_ylim(); axes[0].set_ylim(lo, hi + 0.35 * (hi - lo))
for ax in axes: place_legend(ax, SCALE, italic=False)
save(fig, "fig5_ms_adaptation")
s5.assign(Group=s5.group.map(NAME), Event=s5.event_type.map({"MS": "Multisensory Events", "Control": "Control Events"}))[
    ["Group", "Event", "encounter", "m", "se", "n"]].to_csv(ROOT / "results/figure5_means.csv", index=False)

# ---------------------------------------------------------------- Figure 6: search success
fig = plt.figure(figsize=(15, PANEL_H + 0.9))
gs = GridSpec(1, 3, width_ratios=[1, 1.7, 1.7], wspace=0.30, figure=fig)
a1, a2, a3 = (fig.add_subplot(gs[i]) for i in range(3))
lim = (0, 1.35)
bars(a1, "success_proportion", "Proportion of targets found", lim); age_scatter(a2, "success_proportion", lim)
xl = (0, D.BT.max() * 1.03)
for g in GROUPS:
    x = D[D.group == g]
    ols_band(a3, x.BT, x.success_proportion, COL[g], xl, COLL[g])
    a3.scatter(x.BT, x.success_proportion, s=PT, color=COLD[g], alpha=0.85, lw=0, zorder=3, label=NAME[g])
a3.set_xlim(*xl); a3.set_ylim(0, 1.6); a3.set_xlabel("Backtracking duration (s)")   # headroom for the legend
xinger_axes(a3, SCALE); place_legend(a3, SCALE, italic=False)
for ax in (a1, a2, a3): ax.set_yticks(np.round(np.arange(0, 1.01, 0.2), 1))
save(fig, "fig6_search")

# ---------------------------------------------------------------- Figure S1: regression scatterplots
# Each panel: one outcome against one predictor, with each age group's OLS line and 95% CI from the single-predictor model
# (outcome ~ predictor x age group) on the outcome's common sample: the participants flagged by the outlier rule in the full
# model (common.common_sample, as in 06_model_comparison_common_sample.py) are left out of the fit and drawn as open circles.
ROWS = [("spatial_coverage", "Spatial coverage"), ("opt1", "Object coverage"), ("success_proportion", "Proportion of targets found")]
COLS = [("BT", "Backtracking duration (s)"), ("adaptation", "Adaptation slope (s per encounter)")]
fig, axes = plt.subplots(3, 2, figsize=(10.5, 3 * PANEL_H + 2.2), gridspec_kw={"wspace": 0.18, "hspace": 0.40}, sharex="col", sharey="row")
for i, (y, ylab) in enumerate(ROWS):
    _, X_all, ex = common_sample(D.assign(group=pd.Categorical(D.group, ["adult", "child"])), y)
    X_all = X_all.assign(excluded=X_all.code.isin(ex)); X = X_all[~X_all.excluded]
    for j, (x, xlab) in enumerate(COLS):
        ax = axes[i, j]
        m = smf.ols(f"{y} ~ {x} * group", X).fit()
        xl = (X_all[x].min() - 0.03 * np.ptp(X_all[x]), X_all[x].max() + 0.03 * np.ptp(X_all[x])); xs = np.linspace(*xl, 100)
        for g in ["child", "adult"]:
            new = pd.DataFrame({x: xs, "group": pd.Categorical([g] * len(xs), categories=["adult", "child"])})
            pr = m.get_prediction(new).summary_frame(alpha=0.05)
            ax.fill_between(xs, pr.mean_ci_lower, pr.mean_ci_upper, color=COL[g], alpha=0.2, lw=0)
            ax.plot(xs, pr["mean"], color=COLL[g])
            k, e = X_all[(X_all.group == g) & ~X_all.excluded], X_all[(X_all.group == g) & X_all.excluded]
            ax.scatter(k[x], k[y], s=PT, color=COLD[g], alpha=0.85, lw=0, zorder=3, label=NAME[g])
            ax.scatter(e[x], e[y], s=PT, facecolors="none", edgecolors=COLD[g], lw=1.5 * SCALE, zorder=3)
        ax.set_xlim(*xl)
        if j == 0: ax.set_ylabel(ylab)
        if i == 2: ax.set_xlabel(xlab)
for i, (y, _) in enumerate(ROWS):
    lo, hi = D[y].min(), D[y].max(); axes[i, 0].set_ylim(lo - 0.05 * (hi - lo), hi + 0.45 * (hi - lo))
for ax in axes.flat: xinger_axes(ax, SCALE)
for ax in axes[2]: ax.set_yticks([t for t in ax.get_yticks() if t <= 1.0 + 1e-9])   # proportions stop at 1
place_legend(axes[0, 0], SCALE, italic=False)
align_panel_labels([(ax, l) for ax, l in zip(axes.flat, "abcdef")], SCALE)
save(fig, "figS1_regressions")
print("saved:", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))
