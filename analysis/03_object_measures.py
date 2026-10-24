"""Supplement: object measures compared.

Object measures
  all objects        count of objects looked at (separate from, but correlated with, spatial coverage)
  Option 1           share of the catalogue objects in the grid cells entered that were looked at (main measure)
  Option 1 + spatial Option 1 with spatial coverage as a covariate
  Option 2           share of the catalogue objects in the rooms entered that were looked at
  Option 1, 300 ms   Option 1 with a look counted only if one gaze episode lasted > 300 ms
  all objects, 300 ms
Each goes through the same analyses as the main measure; p values are after the outlier rule,
[in brackets] without exclusions.
Outputs: results/object_measures.txt, results/object_measures.csv, results/object_measures.json
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

(RESULTS / "object_measures.txt").write_text("\n".join(log.lines) + "\n")
import json
json.dump(FULL, open(RESULTS / "object_measures.json", "w"), indent=1, default=float)
