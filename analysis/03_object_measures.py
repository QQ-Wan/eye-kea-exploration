"""Supplement: object measures compared, and robustness to the number of encounters.

Object measures
  all objects        count of objects looked at (separate from, but correlated with, spatial coverage)
  Option 1           share of the catalogue objects in the grid cells entered that were looked at (main measure)
  Option 1 + spatial Option 1 with spatial coverage as a covariate
  Option 2           share of the catalogue objects in the rooms entered that were looked at
  Option 1, 300 ms   Option 1 with a look counted only if one gaze episode lasted > 300 ms
  all objects, 300 ms
Each goes through the same analyses as the main measure; p values are after the outlier rule,
[in brackets] without exclusions.
Encounter ranges: adaptation slopes from the first 3, 4 (main), 5, 6 or all encounters.
Outputs: results/object_measures.txt, results/object_measures.csv, results/encounter_ranges_participant.csv
"""
import sys
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, scipy.stats as st, statsmodels.formula.api as smf
from common import *

MEASURES = [("object_count", "all objects", ""), ("opt1", "Option 1 (main)", ""), ("opt1", "Option 1 + spatial coverage", " + spatial_coverage"),
            ("opt2", "Option 2", ""), ("opt1_300", "Option 1, 300 ms", ""), ("object_count_300_single", "all objects, 300 ms", "")]
log = Log()

def p_pair(D, f, terms, need):
    m, m_all, ex = fit_rule(D, f, need)
    return lin(m, terms), lin(m_all, terms), ex

FULL = {}   # every estimate in full, saved to object_measures.json
def cell(D, f, terms, need, key=None):
    r, ra, ex = p_pair(D, f, terms, need)
    if key: FULL[key] = dict(rule=r, all_data=ra, excluded=ex)
    return f"b={r['b']:+.4f} [{r['CI'][0]:+.4f}, {r['CI'][1]:+.4f}], t({r['df']:.0f})={r['t']:+.2f}, p={r['p']:.3f} [no excl. p={ra['p']:.3f}]"

# ------------------------------------------------------------------ measures side by side (encounters 1-4)
D, E = load()
rows = {}
for y, lab, cov in MEASURES:
    col = {}
    A, C = D[D.group == "adult"], D[D.group == "child"]
    ra_, pa_ = st.pearsonr(A[y], A.spatial_coverage); rc_, pc_ = st.pearsonr(C[y], C.spatial_coverage)
    FULL[f"{lab}|r_spatial"] = dict(adults=dict(r=ra_, df=len(A) - 2, p=pa_), children=dict(r=rc_, df=len(C) - 2, p=pc_))
    col["r with spatial coverage (adults / children)"] = f"r({len(A)-2}) = {ra_:+.2f}, p = {pa_:.3f} / r({len(C)-2}) = {rc_:+.2f}, p = {pc_:.3g}"
    if cov:
        m = smf.ols(f"{y} ~ group{cov}", D).fit(); ci = m.conf_int().loc["group" + CH]
        FULL[f"{lab}|group"] = dict(b=-m.params["group" + CH], CI=[-ci[1], -ci[0]], t=-m.tvalues["group" + CH], df=m.df_resid, p=m.pvalues["group" + CH])
        col["adults vs children"] = (f"controlling spatial coverage: adults - children b={-m.params['group' + CH]:+.4f} [{-ci[1]:+.4f}, {-ci[0]:+.4f}], "
                                     f"t({m.df_resid:.0f})={-m.tvalues['group' + CH]:.2f}, p={m.pvalues['group' + CH]:.3f}")
    else:
        r = ttest(D, y); FULL[f"{lab}|group"] = r
        col["adults vs children"] = (f"adults M={r['adult_M']:.3f}, SD={r['adult_SD']:.3f}; children M={r['child_M']:.3f}, SD={r['child_SD']:.3f}; "
                                     f"t({r['df']:.0f})={r['t']:.2f}, p={r['p']:.3f}, d={r['d']:.2f}")
    r = reg_rule(C, f"{y} ~ age{cov}", "age"); FULL[f"{lab}|age_children"] = r
    col["age effect, children"] = (f"b={r['b']:+.4f} [{r['CI'][0]:+.4f}, {r['CI'][1]:+.4f}], t({r['df']:.0f})={r['t']:+.2f}, p={r['p']:.3f} "
                                   f"[no excl. p={r['all_data']['p']:.3f}]")
    for x, xl in [("BT", "backtracking"), ("adaptation", "adaptation")]:
        f = f"{y} ~ {x}*group{cov}"
        col[f"{xl} -> measure: children"] = cell(D, f, [x, f"{x}:group{CH}"], [y, x], f"{lab}|{xl}|children")
        col[f"{xl} -> measure: adults"] = cell(D, f, [x], [y, x], f"{lab}|{xl}|adults")
        col[f"{xl} -> measure: interaction"] = cell(D, f, [f"{x}:group{CH}"], [y, x], f"{lab}|{xl}|interaction")
    T, picks, chosen = select_model(D, y, cov)
    col["best joint model (AIC, BIC, adj R2 -> chosen)"] = f"{picks['AIC']}, {picks['BIC']}, {picks['adjR2']} -> model {chosen}: {rhs(MODELS[chosen - 2])}"
    if lab == "all objects":
        T.round(4).to_csv(RESULTS / "model_comparison_object_all.csv", index=False)
    rows[lab] = col
O = pd.DataFrame(rows); O.to_csv(RESULTS / "object_measures.csv")
log("=== OBJECT MEASURES (encounters 1-4; p after outlier rule [no exclusions]) ===")
for lab in O.columns:
    log(f"\n{lab}"); [log(f"   {k:48s} {v}") for k, v in O[lab].items()]

# ------------------------------------------------------------------ encounter ranges
log("\n=== ENCOUNTER RANGES: measure ~ adaptation x group; children's slope and the interaction (p after rule [no exclusions]) ===")
R = []
for rl, cap in ENCOUNTER_RANGES.items():
    Dk, _ = load(cap)
    row = dict(encounters=rl)
    for y, lab, cov in MEASURES[:5] + [("spatial_coverage", "spatial coverage", ""), ("success_proportion", "search success", "")]:
        f = f"{y} ~ adaptation*group{cov}"
        c, ca, _ = p_pair(Dk, f, ["adaptation", f"adaptation:group{CH}"], [y, "adaptation"])
        i, ia, _ = p_pair(Dk, f, [f"adaptation:group{CH}"], [y, "adaptation"])
        FULL[f"range {rl}|{lab}|children"] = dict(rule=c, all_data=ca); FULL[f"range {rl}|{lab}|interaction"] = dict(rule=i, all_data=ia)
        row[f"{lab}: children"] = f"{c['p']:.3f} [{ca['p']:.3f}]"; row[f"{lab}: interaction"] = f"{i['p']:.3f} [{ia['p']:.3f}]"
    x = Dk.dropna(subset=["BT", "adaptation", "age", "n_ms_events"])
    for g, xg in [("adults", x[x.group == "adult"]), ("children", x[x.group == "child"]), ("all", x)]:
        r = partial_r(xg); row[f"backtracking-adaptation r, {g}"] = f"{r['r']:.2f} (p={r['p']:.3f})"; FULL[f"range {rl}|partial_r|{g}"] = r
    R.append(row)
R = pd.DataFrame(R).set_index("encounters").T
R.to_csv(RESULTS / "encounter_ranges_participant.csv")
log(R.to_string())
(RESULTS / "object_measures.txt").write_text("\n".join(log.lines) + "\n")
import json
json.dump(FULL, open(RESULTS / "object_measures.json", "w"), indent=1, default=float)
