"""Regressions of spatial coverage, object coverage and search success on backtracking and adaptation (Tables S2-S5).

For each outcome:
  common sample  - the full model (model 19) is fit to all participants; the participants it flags under the outlier rule
                   (Cook's D > 4/n AND |externally studentized residual| > 2.5) are excluded once, and backtracking and adaptation
                   are centered at the mean of the remaining sample (common.common_sample). Every model below uses this sample.
  model comparison (Tables S2-S4) - model 1 (intercept only) and models 2-19 (the 18 combinations of backtracking, adaptation,
                   age group and their interactions). The selected model is the one favored by both AIC and BIC; when they disagree,
                   the model better by more than 2 on one criterion and within 2 on the other (common.select_aic_bic).
  single-predictor models - outcome ~ backtracking x age group and outcome ~ adaptation x age group, with each age group's slope.
Each estimate is also given for all participants (Table S5). The age regressions in Table S5 apply the outlier rule to each model
separately and are taken from 01_participant_stats.py (run it first).

Outputs in results/model_comparison_common_sample/:
  summary.txt         all results, formatted as in the manuscript
  tables_S2_S4.md     Tables S2-S4 (and tables_S2_S4.csv)
  table_S5.md         Table S5 (and table_S5.csv)
  results.json        every estimate at full precision
Supplementary Figure S1 (the single-predictor models on the common samples) is drawn by 05_figures.py.
"""
import json
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from common import load, common_sample, select_aic_bic, model_formula, SPECS, lin, CH
from config import RESULTS

OUT = RESULTS / "model_comparison_common_sample"
OUT.mkdir(exist_ok=True)
D, _ = load()
OUTCOMES = [("spatial_coverage", "Spatial Coverage"), ("opt1", "Object Coverage"), ("success_proportion", "Target Search Success Rate")]
NAMES = {"B": "Total Backtracking Duration", "A": "Adaptation to Multisensory Events", "G": "Age Group"}
PREDICTORS = [("BT_c", "backtracking"), ("adaptation_c", "adaptation")]

def label(i): return " + ".join(" × ".join(NAMES[t] for t in s.split(":")) for s in SPECS[i - 1])
def coef(m, k): return dict(b=m.params[k], CI=list(m.conf_int().loc[k]), t=m.tvalues[k], df=m.df_resid, p=m.pvalues[k], n=int(m.nobs))
def fit_summary(m):
    coefs = {k: coef(m, k) for k in m.params.index if k != "Intercept"}
    slopes = {}
    for pred, nm in PREDICTORS:
        if pred not in m.params: continue
        inter = f"{pred}:group{CH}"
        if inter in m.params: slopes[f"children_{nm}"] = lin(m, [pred, inter]); slopes[f"adults_{nm}"] = lin(m, [pred])
        else: slopes[f"both_groups_{nm}"] = lin(m, [pred])   # no interaction with age group: one slope for both groups
    return dict(n=int(m.nobs), R2=m.rsquared, adjR2=m.rsquared_adj, F=m.fvalue, F_df=[m.df_model, m.df_resid], F_p=m.f_pvalue,
                coefs=coefs, slopes=slopes)
def moderated(X, y, pred):
    m = smf.ols(f"{y} ~ {pred} * group", X).fit(); inter = f"{pred}:group{CH}"
    return dict(interaction=coef(m, inter), children=lin(m, [pred, inter]), adults=lin(m, [pred]), n=int(m.nobs))

# ------------------------------------------------------------------------------------------------------------------------ run
res = {}
for y, title in OUTCOMES:
    X, X_all, ex = common_sample(D, y)
    T = pd.DataFrame([dict(model=i, k=int(m.df_model) + 1, n=int(m.nobs), AIC=m.aic, BIC=m.bic,
                           formula="Intercept only" if i == 1 else f"{title} = {label(i)}")
                      for i in range(1, 20) for m in [smf.ols(model_formula(y, i), X).fit()]])
    T["dAIC"], T["dBIC"] = T.AIC - T.AIC.min(), T.BIC - T.BIC.min()
    sel = select_aic_bic(T); i = sel["chosen"]
    assert i is not None, f"{y}: AIC and BIC do not decide"
    res[y] = dict(title=title, n=len(X), excluded=ex, table=T.to_dict("records"), selection=dict(sel, formula=label(i)),
                  selected=dict(common=fit_summary(smf.ols(model_formula(y, i), X).fit()),
                                all_data=fit_summary(smf.ols(model_formula(y, i), X_all).fit())),
                  single={nm: dict(common=moderated(X, y, pred), all_data=moderated(X_all, y, pred)) for pred, nm in PREDICTORS})
json.dump(res, open(OUT / "results.json", "w"), indent=1, default=float)

# ------------------------------------------------------------------------------------------------------------------------ formatting
MINUS = "−"
def num(x, d): return f"{(0.0 if round(x, d) == 0 else x):.{d}f}".replace("-", MINUS)   # no "−0.000"
def nz(x, d):
    s = f"{x:.{d}f}"; s = s.replace("0.", ".", 1) if s.startswith(("0.", "-0.")) else s; return s.replace("-", MINUS)
def P(p): return "p < .001" if p < .001 else "p > .999" if p > .999 else f"p = {nz(p, 3)}"
def dec_for(*xs):
    m = max(abs(x) for x in xs)
    for lim, d in [(10, 2), (1, 2), (.1, 3), (.01, 3), (.001, 4)]:
        if m >= lim: return d
    return 5
def B(r):
    d = dec_for(r["b"], *r["CI"])
    while r["b"] != 0 and round(abs(r["b"]), d) == 0 and d < 6: d += 1     # at least one significant digit of the estimate
    return f"β = {num(r['b'], d)}, 95% CI [{num(r['CI'][0], d)}, {num(r['CI'][1], d)}], t({int(round(r['df']))}) = {num(r['t'], 2)}, {P(r['p'])}"
def FIT(f): return (f"R² = {nz(f['R2'], 3)} ({f['R2'] * 100:.1f}% of the variance), adjusted R² = {nz(f['adjR2'], 3)}, "
                    f"F({int(f['F_df'][0])}, {int(f['F_df'][1])}) = {num(f['F'], 2)}, {P(f['F_p'])}")
def grp(codes): return ", ".join(f"{c} ({'child' if c.startswith('C') else 'adult'})" for c in codes) or "none"
TERM = {"group[T.child]": "age group (children vs adults)", "BT_c": "backtracking", "adaptation_c": "adaptation",
        "BT_c:adaptation_c": "backtracking × adaptation", "BT_c:group[T.child]": "backtracking × age group",
        "adaptation_c:group[T.child]": "adaptation × age group", "BT_c:adaptation_c:group[T.child]": "backtracking × adaptation × age group"}

# ------------------------------------------------------------------------------------------------------------------------ summary.txt
L = ["REGRESSIONS ON BACKTRACKING AND ADAPTATION (one common sample per outcome; selection by AIC and BIC)", ""]
for y, r in res.items():
    s, f = r["selection"], r["selected"]["common"]
    L += [f"==================== {r['title'].upper()} ====================",
          f"common sample: n = {r['n']}; excluded by the outlier rule in the full model: {grp(r['excluded'])}",
          f"AIC picks model {s['AIC_pick']}, BIC picks model {s['BIC_pick']}: {s['how']}",
          f"selected model {s['chosen']}: {s['formula']}", f"   {FIT(f)}"]
    L += [f"   {TERM.get(k, k):40s} {B(c)}" for k, c in f["coefs"].items()]
    L += [f"   simple slope, {k.replace('_', ' '):30s} {B(c)}" for k, c in f["slopes"].items()]
    L.append(f"   all participants: {FIT(r['selected']['all_data'])}")
    for nm in ["backtracking", "adaptation"]:
        c = r["single"][nm]["common"]
        L.append(f"single predictor, {nm} × age group: interaction {B(c['interaction'])}; children {B(c['children'])}; adults {B(c['adults'])}")
    L.append("")
(OUT / "summary.txt").write_text("\n".join(L))
print("\n".join(L))

# ------------------------------------------------------------------------------------------------------------------------ Tables S2-S4
CAPTION = {"spatial_coverage": "Model Comparison for Spatial Coverage Analysis",
           "opt1": "Model Comparison for Object Coverage Analysis (share of objects in the visited cells that were looked at)",
           "success_proportion": "Model Comparison for Target Search Success Rate Analysis"}
md, tab = [], []
for n_, (y, r) in enumerate(res.items(), start=2):
    s = r["selection"]; T = pd.DataFrame(r["table"])
    nc = sum(c.startswith("C") for c in r["excluded"]); na = len(r["excluded"]) - nc
    md += [f"### Table S{n_}. {CAPTION[y]}", "", "| Model | Param No. | AIC | BIC | ΔAIC | ΔBIC | Formula |", "|---|---|---|---|---|---|---|"]
    md += [(f"| **{w.model}** | **{w.k}** | **{num(w.AIC, 2)}** | **{num(w.BIC, 2)}** | **{num(w.dAIC, 2)}** | **{num(w.dBIC, 2)}** | **{w.formula}** |"
            if w.model == s["chosen"] else
            f"| {w.model} | {w.k} | {num(w.AIC, 2)} | {num(w.BIC, 2)} | {num(w.dAIC, 2)} | {num(w.dBIC, 2)} | {w.formula} |") for w in T.itertuples()]
    note = (f"Note. All models were fit to the same participants (n = {r['n']}; excluded by the outlier rule in the full model: "
            f"{nc} {'child' if nc == 1 else 'children'}, {na} {'adult' if na == 1 else 'adults'}). Backtracking and adaptation were centered at "
            f"the sample mean. The selected model (bold) was favored by both AIC and BIC, or, when they disagreed, was better by more than 2 on "
            f"one criterion and within 2 on the other.")
    if s["AIC_pick"] != s["BIC_pick"]:
        note += f" AIC favored model {s['AIC_pick']} by {s['dAIC']:.2f}, and BIC favored model {s['BIC_pick']} by {s['dBIC']:.2f}."
    md += ["", note, ""]
    tab.append(T.assign(table=f"S{n_}", outcome=r["title"], selected=T.model == s["chosen"]))
(OUT / "tables_S2_S4.md").write_text("\n".join(md))
pd.concat(tab)[["table", "outcome", "model", "k", "n", "AIC", "BIC", "dAIC", "dBIC", "selected", "formula"]].to_csv(OUT / "tables_S2_S4.csv", index=False)

# ------------------------------------------------------------------------------------------------------------------------ Table S5
S, O, Q = res["spatial_coverage"], res["opt1"], res["success_proportion"]
assert (S["selection"]["chosen"], O["selection"]["chosen"], Q["selection"]["chosen"]) == (10, 3, 17), "selected models changed: update the rows"
rows = []
def add(analysis, r, common, all_data): rows.append(dict(analysis=analysis, excluded=len(r["excluded"]), common_sample=B(common), all_participants=B(all_data)))
def sel(r, key, kind="coefs"): return r["selected"]["common"][kind][key], r["selected"]["all_data"][kind][key]
add("Spatial coverage, selected model: backtracking × age group", S, *sel(S, "BT_c:group[T.child]"))
add("Spatial coverage, selected model: backtracking, children", S, *sel(S, "children_backtracking", "slopes"))
add("Spatial coverage, selected model: backtracking, adults", S, *sel(S, "adults_backtracking", "slopes"))
for r, oname in [(S, "spatial coverage"), (O, "object coverage")]:
    for nm in ["backtracking", "adaptation"]:
        c, a = r["single"][nm]["common"], r["single"][nm]["all_data"]
        add(f"{nm.capitalize()} → {oname}, children", r, c["children"], a["children"])
        add(f"{nm.capitalize()} → {oname}, adults", r, c["adults"], a["adults"])
        add(f"{nm.capitalize()} × age group → {oname}", r, c["interaction"], a["interaction"])
add("Object coverage, selected model: adaptation", O, *sel(O, "adaptation_c"))
add("Search success, selected model: backtracking × age group", Q, *sel(Q, "BT_c:group[T.child]"))
add("Search success, selected model: adaptation × age group", Q, *sel(Q, "adaptation_c:group[T.child]"))
for key, lab in [("children_backtracking", "backtracking, children"), ("adults_backtracking", "backtracking, adults"),
                 ("children_adaptation", "adaptation, children"), ("adults_adaptation", "adaptation, adults")]:
    add(f"Search success, selected model: {lab}", Q, *sel(Q, key, "slopes"))
# age regressions with exclusions (outlier rule applied to each model; 01_participant_stats.py)
PS = json.load(open(RESULTS / "participant_stats.json"))
AGE_LAB = [("spatial_coverage", "spatial coverage"), ("object_count", "objects looked at"), ("opt1", "object coverage"), ("BT", "backtracking duration"),
           ("BT_distance", "backtracking distance"), ("success", "search success"), ("success_proportion", "search success"),
           ("rt", "search time"), ("average_success_rt", "search time"), ("adaptation", "adaptation")]
for pre, g in [("age_", "children"), ("adult_age_", "adults")]:
    for v, lab in AGE_LAB:
        r_ = PS.get(pre + v)
        if r_ is not None and r_["excluded"]: add(f"Age → {lab}, {g}", r_, r_, r_["all_data"])
assert max(x["excluded"] for x in rows) <= 2
T5 = pd.DataFrame(rows); T5.to_csv(OUT / "table_S5.csv", index=False)
(OUT / "table_S5.md").write_text("\n".join(
    ["### Table S5. Main Results on the Common Sample and With All Participants", "",
     "| Analysis | Excluded | Common sample | All participants |", "|---|---|---|---|"]
    + [f"| {w.analysis} | {w.excluded} | {w.common_sample} | {w.all_participants} |" for w in T5.itertuples()]
    + ["", "Note. For the models relating backtracking and adaptation to each outcome, outliers were identified once per outcome in the full "
           "model (Cook’s distance > 4/n and |studentized residual| > 2.5). Age regressions applied the same rule to each model separately.", ""]))
print("written:", ", ".join(sorted(p.name for p in OUT.iterdir())))
