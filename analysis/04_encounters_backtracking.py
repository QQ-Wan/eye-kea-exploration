"""Supplement: repeated encounters with multisensory (MS) objects and backtracking.
(1) participant level: Spearman correlations of backtracking with encounter counts;
(2) event level: share of MS events on the return leg of a backtracking episode, by encounter;
(3) whether a backtracking turn follows MS events more often than control events (within 10 s).
Output: results/encounters_backtracking.txt
"""
import sys
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, scipy.stats as st, statsmodels.formula.api as smf
from common import *

EP = pd.read_csv(DATA / "backtracking_episodes.csv")
E = pd.read_csv(DATA / "events.csv"); E["time"] = E.window_start + 5.0      # event onset (the pre window starts 5 s before)
S = pd.read_csv(DATA / "subject_measures.csv").set_index("code")
log = Log()

# ---- (1) participant level
ms = E[E.event_type == "MS"]
pp = pd.DataFrame({"n_ms": ms.groupby("code").size(),
                   "n_repeat": ms[ms.encounter >= 2].groupby("code").size(),
                   "max_enc": ms.groupby("code").encounter.max(),
                   "mean_enc_per_obj": ms.groupby(["code", "object"]).size().groupby("code").mean(),
                   "n_objects": ms.groupby("code")["object"].nunique()}).fillna(0)
pp = pp.join(S[["group", "BT", "spatial_coverage"]])
log("=== (1) Participant level: Spearman correlations with backtracking duration ===")
for g in ["all", "adult", "child"]:
    x = pp if g == "all" else pp[pp.group == g]
    cells = []
    for v in ["n_ms", "n_repeat", "mean_enc_per_obj", "max_enc", "n_objects"]:
        r, p = st.spearmanr(x.BT, x[v]); cells.append(f"{v} r={r:+.2f} (p={p:.3f})")
    log(f"{g:6s} (n={len(x)}): " + "; ".join(cells))

# ---- (2) event level: is the event on the outbound leg (start -> turn) or the return leg (turn -> end) of an episode?
def phase(row):
    e = EP[(EP.code == row.code) & (EP.block == row.block)]
    if ((e.turn <= row.time) & (row.time <= e.end)).any(): return "return"
    if ((e.start <= row.time) & (row.time < e.turn)).any(): return "outbound"
    return "outside"
E["phase"] = [phase(r) for r in E.itertuples()]
log("\n=== (2) Share of MS events on each leg of a backtracking episode, by encounter ===")
ms = E[E.event_type == "MS"].copy(); ms["enc_cat"] = np.where(ms.encounter >= 3, "3+", ms.encounter.astype(int).astype(str))
for g in ["adult", "child"]:
    x = ms[ms.group == g]
    tab = x.groupby("enc_cat").phase.value_counts(normalize=True).unstack(fill_value=0); n = x.groupby("enc_cat").size()
    log(f"{g}: " + " | ".join(f"encounter {k}: return {tab.loc[k].get('return', 0):.1%}, outbound {tab.loc[k].get('outbound', 0):.1%}, outside {tab.loc[k].get('outside', 0):.1%} (n={n[k]})" for k in tab.index))
ms["is_return"] = (ms.phase == "return").astype(int); ms["repeat"] = (ms.encounter >= 2).astype(int)
for g in ["adult", "child"]:
    x = ms[ms.group == g]
    m = smf.logit("is_return ~ repeat", x).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(x.code)[0]})
    ci = np.exp(m.conf_int().loc["repeat"])
    log(f"   {g}: repeat (vs first) encounter -> odds of being on a return leg: OR = {np.exp(m.params['repeat']):.2f}, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}], "
        f"z = {m.tvalues['repeat']:.2f}, p = {m.pvalues['repeat']:.2g} ({len(x)} MS events, {x.code.nunique()} participants; SE clustered by participant)")

# ---- (3) does a backtracking turn follow the event within 10 s? MS vs control
log("\n=== (3) Share of events followed by a backtracking turn within 10 s ===")
def turn_after(row, w=10.0):
    e = EP[(EP.code == row.code) & (EP.block == row.block)]
    return bool(((e.turn > row.time) & (e.turn <= row.time + w)).any())
E["turn_10s"] = [turn_after(r) for r in E.itertuples()]
for g in ["adult", "child"]:
    x = E[E.group == g].assign(ms=lambda d: (d.event_type == "MS").astype(int), turn=lambda d: d.turn_10s.astype(int))
    m = smf.logit("turn ~ ms", x).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(x.code)[0]})
    ci = np.exp(m.conf_int().loc["ms"])
    log(f"{g}: MS {x[x.ms == 1].turn.mean():.1%} ({int(x[x.ms == 1].turn.sum())}/{int((x.ms == 1).sum())}) vs control {x[x.ms == 0].turn.mean():.1%} "
        f"({int(x[x.ms == 0].turn.sum())}/{int((x.ms == 0).sum())}); OR = {np.exp(m.params['ms']):.2f}, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}], z = {m.tvalues['ms']:.2f}, p = {m.pvalues['ms']:.3f}")
(RESULTS / "encounters_backtracking.txt").write_text("\n".join(log.lines) + "\n")
