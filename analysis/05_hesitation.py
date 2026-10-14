"""Outbound vs return legs of backtracking episodes: pauses and steps.

For every backtracking episode, the outbound leg (start -> turning point) is compared with the
return leg (turning point -> end):
  pause = time between consecutive teleports within the leg
  step  = distance covered by a teleport (teleports with no other teleport between two head samples)
  rate  = teleports per second
Participant level: median pause and median step per leg type, pooled over episodes; the group x leg
interaction is tested on the return-minus-outbound differences.
Output: results/hesitation.txt
"""
import sys
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, scipy.stats as st
from common import *

EP = pd.read_csv(DATA / "backtracking_episodes.csv")
TP = pd.read_csv(DATA / "teleports.csv").sort_values(["code", "block", "time"])
group = pd.read_csv(DATA / "participants.csv").set_index("code").group
rows = []
for (code, blk), eps in EP.groupby(["code", "block"]):
    tp = TP[(TP.code == code) & (TP.block == blk)]
    for i, e in enumerate(eps.itertuples()):
        for leg, t0, t1 in [("outbound", e.start, e.turn), ("return", e.turn, e.end)]:
            tl = tp[(tp.time >= t0) & (tp.time < t1)]
            rows.append(dict(code=code, group=group[code], episode=f"{blk}_{i}", leg=leg, duration=t1 - t0, n_teleports=len(tl),
                             pauses=list(np.diff(tl.time.values)), steps=tl.step.dropna().tolist()))
L = pd.DataFrame(rows)
def pooled_median(lists):
    v = sum(lists, []); return np.median(v) if v else np.nan
P = (L.groupby(["code", "group", "leg"])
       .apply(lambda x: pd.Series(dict(pause=pooled_median(list(x.pauses)), step=pooled_median(list(x.steps)),
                                       rate=x.n_teleports.sum() / x.duration.sum())))
       .reset_index())
W = P.pivot_table(index=["code", "group"], columns="leg", values=["pause", "step", "rate"]).dropna()
log = Log()
log(f"backtracking episodes: {L.episode.groupby(L.code).nunique().sum()} from {W.shape[0]} participants with both leg types "
    f"(adults {int((W.index.get_level_values(1) == 'adult').sum())}, children {int((W.index.get_level_values(1) == 'child').sum())})")
for var, unit, lab in [("pause", "s", "median pause between teleports"), ("step", "m", "median step per teleport"), ("rate", "/s", "teleports per second")]:
    log(f"\n=== {lab} ===")
    diffs = {}
    for g in ["adult", "child"]:
        x = W.xs(g, level=1); o, r = x[(var, "outbound")], x[(var, "return")]
        t, p = st.ttest_rel(r, o); diffs[g] = r - o
        log(f"{g:6s} (n={len(x)}): outbound {o.mean():.3f} {unit}, return {r.mean():.3f} {unit}; return - outbound = {(r - o).mean():+.3f} (paired t = {t:.2f}, p = {p:.3f})")
    t, p = st.ttest_ind(diffs["child"], diffs["adult"], equal_var=False)
    log(f"group x leg (return - outbound, children vs adults): Welch t = {t:.2f}, p = {p:.3f}")
    for leg in ["outbound", "return"]:
        a_, c_ = W.xs("adult", level=1)[(var, leg)], W.xs("child", level=1)[(var, leg)]
        t, p = st.ttest_ind(c_, a_, equal_var=False)
        log(f"   {leg:8s} leg, children vs adults: {c_.mean():.3f} vs {a_.mean():.3f}, Welch t = {t:.2f}, p = {p:.3f}")
(RESULTS / "hesitation.txt").write_text("\n".join(log.lines) + "\n")
