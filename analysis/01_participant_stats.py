"""Participant-level statistics: group differences, age effects, regressions of exploration and
search outcomes on backtracking and adaptation, model comparison, and the reviewer-requested
additions (age effects within adults, VR and video-game experience).

Object coverage (opt1): the share of catalogue objects in the grid cells a participant entered that
they looked at. The count of all objects looked at is reported descriptively.
Outputs: results/participant_stats.{txt,json}, results/model_comparison_*.csv
"""
import sys
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, scipy.stats as st, statsmodels.formula.api as smf
from statsmodels.stats.power import TTestIndPower
from common import *

D, E = load()
A, C = D[D.group == "adult"], D[D.group == "child"]
log = Log()

# ------------------------------------------------------------------ participants
log("=== PARTICIPANTS ===")
log.put("n_adults", len(A)); log.put("n_children", len(C))
log(f"adults n={len(A)}, children n={len(C)}; children by year of age: {C.age.astype(int).value_counts().sort_index().to_dict()}")
log(f"adults: age M={A.age.mean():.2f}, SD={A.age.std():.2f}, range {A.age.min():.0f}-{A.age.max():.0f}")
pw = TTestIndPower().solve_power(nobs1=len(C), ratio=len(A) / len(C), alpha=.05, power=.8)
z = (st.norm.ppf(.975) + st.norm.ppf(.8)) / np.sqrt(len(D) - 3 - 2)
log.put("sensitivity", dict(d=pw, partial_r=float(np.tanh(z))))
log(f"sensitivity (80% power, alpha .05): two-group t-test d={pw:.2f}; partial correlation with 2 covariates r={np.tanh(z):.2f}")

# ------------------------------------------------------------------ exploration
log("\n=== EXPLORATION ===")
for v in ["spatial_coverage", "object_count", "opt1", "distance_traveled"]:
    r = ttest(D, v); log.put(f"t_{v}", r)
    log(f"{v}: adults {r['adult_M']:.3f} ({r['adult_SD']:.3f}) vs children {r['child_M']:.3f} ({r['child_SD']:.3f}); t({r['df']:.0f})={r['t']:.2f}, p={r['p']:.4f}, Mdiff={r['Mdiff']:.3f} [{r['CI'][0]:.3f}, {r['CI'][1]:.3f}], d={r['d']:.2f}")
for v in ["spatial_coverage", "object_count", "opt1"]:
    r = reg_rule(C, f"{v} ~ age", "age"); log.put(f"age_{v}", r); log(f"  children: {v} ~ age: {fmt(r)} (excluded: {r['excluded'] or 'none'}; all data p={r['all_data']['p']:.4f})")
for v in ["BT", "BT_distance"]:
    r = ttest(D, v, welch=True); log.put(f"t_{v}", r)
    log(f"{v} (Welch): adults {r['adult_M']:.2f} ({r['adult_SD']:.2f}) vs children {r['child_M']:.2f} ({r['child_SD']:.2f}); t({r['df']:.1f})={r['t']:.2f}, p={r['p']:.4f}, Mdiff={r['Mdiff']:.2f} [{r['CI'][0]:.2f}, {r['CI'][1]:.2f}], d={r['d']:.2f}")
    r = reg_rule(C, f"{v} ~ age", "age"); log.put(f"age_{v}", r); log(f"  children: {v} ~ age: {fmt(r)} (excluded: {r['excluded'] or 'none'}; all data p={r['all_data']['p']:.4f})")
r = ttest(D, "step_median", welch=True); log.put("t_step_median", r)
log(f"teleport step (participant median, m): adults {r['adult_M']:.2f} ({r['adult_SD']:.2f}) vs children {r['child_M']:.2f} ({r['child_SD']:.2f}); t({r['df']:.1f})={r['t']:.2f}, p={r['p']:.3f}, d={r['d']:.2f}")
r = ttest(D, "n_teleports", welch=True); log.put("t_n_teleports", r)
log(f"number of teleports: adults {r['adult_M']:.1f} vs children {r['child_M']:.1f}; t({r['df']:.1f})={r['t']:.2f}, p={r['p']:.3f}")

# ------------------------------------------------------------------ search
log("\n=== SEARCH ===")
r = ttest(D, "success_proportion"); log.put("t_success", r)
log(f"success: adults {r['adult_M']:.3f} ({r['adult_SD']:.3f}) vs children {r['child_M']:.3f} ({r['child_SD']:.3f}); t({r['df']:.0f})={r['t']:.2f}, p={r['p']:.4f}, Mdiff={r['Mdiff']:.3f} [{r['CI'][0]:.3f}, {r['CI'][1]:.3f}], d={r['d']:.2f}")
r = reg_rule(C, "success_proportion ~ age", "age"); log.put("age_success", r); log(f"  children: success ~ age: {fmt(r)} (excluded: {r['excluded'] or 'none'}; all data p={r['all_data']['p']:.4f})")
r = ttest(D, "average_success_rt"); log.put("t_rt", r)
log(f"search time: adults {r['adult_M']:.2f} ({r['adult_SD']:.2f}) vs children {r['child_M']:.2f} ({r['child_SD']:.2f}); t({r['df']:.0f})={r['t']:.2f}, p={r['p']:.4f}, Mdiff={r['Mdiff']:.2f} [{r['CI'][0]:.2f}, {r['CI'][1]:.2f}], d={r['d']:.2f}")
r = reg_rule(C, "average_success_rt ~ age", "age"); log.put("age_rt", r); log(f"  children: search time ~ age: {fmt(r)} (excluded: {r['excluded'] or 'none'}; all data p={r['all_data']['p']:.4f})")

# ------------------------------------------------------------------ one predictor at a time
log("\n=== MODELS WITH ONE PREDICTOR (outlier rule applied per model; [all data] = no exclusions) ===")
for y in ["spatial_coverage", "opt1", "success_proportion"]:
    for x in ["BT", "adaptation"]:
        m, m_all, ex = fit_rule(D, f"{y} ~ {x}*group", [y, x])
        res = dict(excluded=ex, children=lin(m, [x, f"{x}:group{CH}"]), adults=lin(m, [x]), interaction=lin(m, [f"{x}:group{CH}"]),
                   children_all_data=lin(m_all, [x, f"{x}:group{CH}"]), adults_all_data=lin(m_all, [x]), interaction_all_data=lin(m_all, [f"{x}:group{CH}"]))
        log.put(f"single_{y}_{x}", res)
        log(f"{y} ~ {x} x group  (excluded: {ex or 'none'})\n   children    {fmt(res['children'])}\n   adults      {fmt(res['adults'])}\n   interaction {fmt(res['interaction'])}"
            f"\n   [all data] children p={res['children_all_data']['p']:.4f}, adults p={res['adults_all_data']['p']:.4f}, interaction p={res['interaction_all_data']['p']:.4f}")

# ------------------------------------------------------------------ backtracking and adaptation
log("\n=== BACKTRACKING AND ADAPTATION (partial correlation, controlling for age and number of MS events) ===")
x = D.dropna(subset=["BT", "adaptation", "age", "n_ms_events"])
for g, xg in [("adults", x[x.group == "adult"]), ("children", x[x.group == "child"]), ("all participants (supplement)", x)]:
    r = partial_r(xg); log.put(f"partial_r_{g.split()[0]}", r)
    log(f"{g}: r({r['df']}) = {r['r']:.3f}, 95% CI [{r['CI'][0]:.3f}, {r['CI'][1]:.3f}], p = {r['p']:.3f}")

# ------------------------------------------------------------------ model comparison (Tables S1-S3)
log("\n=== MODEL COMPARISON (participants with all predictors; centered predictors; majority of AIC, BIC, adjusted R2) ===")
def compare(y, label):
    T, picks, chosen = select_model(D, y); X = D.dropna(subset=[y, "BT", "adaptation"])
    T.round(4).to_csv(RESULTS / f"model_comparison_{label}.csv", index=False)
    m, m_all, ex = fit_rule(D, f"{y} ~ {rhs(MODELS[chosen - 2])}", [y, "BT", "adaptation"])
    coefs = {k: dict(b=m.params[k], CI=list(m.conf_int().loc[k]), t=m.tvalues[k], df=m.df_resid, p=m.pvalues[k],
                     all_data=dict(b=m_all.params[k], CI=list(m_all.conf_int().loc[k]), t=m_all.tvalues[k], df=m_all.df_resid, p=m_all.pvalues[k]),
                     p_all_data=m_all.pvalues[k]) for k in m.params.index if k != "Intercept"}
    slopes = {}
    for pred, nm in [("BT_c", "backtracking"), ("adaptation_c", "adaptation")]:
        if pred in m.params:
            inter = f"{pred}:group{CH}"
            if inter in m.params:
                slopes[f"children_{nm}"] = lin(m, [pred, inter]); slopes[f"adults_{nm}"] = lin(m, [pred])
            else:
                slopes[f"all_{nm}"] = lin(m, [pred])
    log.put(f"model_{label}", dict(n=len(X), picks=picks, chosen=chosen, formula=rhs(MODELS[chosen - 2]), excluded=ex, adjR2=m.rsquared_adj,
                                   all_data=dict(adjR2=m_all.rsquared_adj, F=m_all.fvalue, F_df=[m_all.df_model, m_all.df_resid], F_p=m_all.f_pvalue),
                                   F=m.fvalue, F_df=[m.df_model, m.df_resid], F_p=m.f_pvalue, coefs=coefs, slopes=slopes))
    log(f"{label} (n={len(X)}): best by AIC={picks['AIC']}, BIC={picks['BIC']}, adjR2={picks['adjR2']} -> model {chosen}: {rhs(MODELS[chosen - 2])}")
    log(f"   excluded: {ex or 'none'}; adj R2={m.rsquared_adj:.3f}, F({m.df_model:.0f},{m.df_resid:.0f})={m.fvalue:.2f}, p={m.f_pvalue:.4f}")
    for k, v in coefs.items():
        log(f"      {k:40s} b={v['b']:+.4f}, t={v['t']:+.2f}, p={v['p']:.4f}  [all data p={v['p_all_data']:.4f}]")
    for k, v in slopes.items():
        log(f"      simple slope {k:24s} {fmt(v)}")
for y, label in [("spatial_coverage", "spatial_coverage"), ("opt1", "object_option1"), ("success_proportion", "search_success")]:
    compare(y, label)

# ------------------------------------------------------------------ reviewer-requested additions
log("\n=== AGE EFFECTS WITHIN ADULTS ===")
for v in ["spatial_coverage", "object_count", "opt1", "BT", "success_proportion", "average_success_rt", "adaptation"]:
    r = reg_rule(A, f"{v} ~ age", "age"); log.put(f"adult_age_{v}", r)
    log(f"  adults: {v} ~ age: t({r['df']:.0f})={r['t']:+.2f}, p={r['p']:.3f} (excluded: {r['excluded'] or 'none'}; all data p={r['all_data']['p']:.3f})")
log("\n=== VR AND VIDEO-GAME EXPERIENCE (1-7, 7 = strongly agree) ===")
for v, label in [("vr_experience", "VR experience"), ("game_experience", "video-game experience")]:
    a, c = A[v].dropna(), C[v].dropna(); t, p = st.ttest_ind(a, c, equal_var=False)
    va, vc = a.var() / len(a), c.var() / len(c); dfw = (va + vc) ** 2 / (va ** 2 / (len(a) - 1) + vc ** 2 / (len(c) - 1))
    log.put(f"q_{v}", dict(adult_M=a.mean(), adult_SD=a.std(), n_adult=len(a), child_M=c.mean(), child_SD=c.std(), n_child=len(c), t=t, df=dfw, p=p,
                           adult_agree=int((a >= 5).sum()), child_agree=int((c >= 5).sum())))
    log(f"  {label}: adults {a.mean():.2f} ({a.std():.2f}, n={len(a)}) vs children {c.mean():.2f} ({c.std():.2f}, n={len(c)}); Welch t={t:.2f}, p={p:.3f}; agree (5-7): adults {int((a>=5).sum())}, children {int((c>=5).sum())}")

log("\n=== AGE PATTERNS OF BACKTRACKING AND ADAPTATION ===")
r = reg_rule(C, "adaptation ~ age", "age"); log.put("age_adaptation", r); log(f"  children: adaptation ~ age: {fmt(r)} (excluded: {r['excluded'] or 'none'}; all data p={r['all_data']['p']:.4f})")
for lo, hi in [(7, 10), (11, 14)]:
    cb = C[C.age.between(lo, hi)]
    for v in ["BT", "adaptation"]:
        a_, c_ = A[v].dropna(), cb[v].dropna(); t, pv = st.ttest_ind(c_, a_, equal_var=False)
        va, vc = a_.var() / len(a_), c_.var() / len(c_); dfw = (va + vc) ** 2 / (va ** 2 / (len(a_) - 1) + vc ** 2 / (len(c_) - 1))
        log.put(f"band_{lo}_{hi}_{v}", dict(child_M=c_.mean(), child_SD=c_.std(), n_child=len(c_), adult_M=a_.mean(), adult_SD=a_.std(), n_adult=len(a_), t=t, df=dfw, p=pv))
        log(f"  exploratory: ages {lo}-{hi} (n={len(c_)}) vs adults, {v}: {c_.mean():.3f} ({c_.std():.3f}) vs {a_.mean():.3f} ({a_.std():.3f}), Welch t({dfw:.1f}) = {t:.2f}, p = {pv:.3f}")
log("\n=== OBJECT COVERAGE AND SPATIAL COVERAGE ===")
for v, key in [("object_count", "r_count_spatial"), ("opt1", "r_opt1_spatial_tmp")]:
    xx = D.dropna(subset=[v, "spatial_coverage"]); rr, prr = st.pearsonr(xx[v], xx.spatial_coverage); zz = np.arctanh(rr); sez = 1 / np.sqrt(len(xx) - 3)
    log.put(key, dict(r=rr, df=len(xx) - 2, p=prr, CI=[np.tanh(zz - 1.96 * sez), np.tanh(zz + 1.96 * sez)], n=len(xx)))
    log(f"{v} vs spatial coverage: r({len(xx)-2}) = {rr:.3f}, 95% CI [{np.tanh(zz-1.96*sez):.3f}, {np.tanh(zz+1.96*sez):.3f}], p = {prr:.2g}")
x = D.dropna(subset=["opt1", "spatial_coverage"]); r, pr = st.pearsonr(x.opt1, x.spatial_coverage); z = np.arctanh(r); se = 1 / np.sqrt(len(x) - 3)
log.put("r_opt1_spatial", dict(r=r, df=len(x) - 2, p=pr, CI=[np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)], n=len(x)))
log(f"object coverage vs spatial coverage, all participants: r({len(x)-2}) = {r:.3f}, 95% CI [{np.tanh(z-1.96*se):.3f}, {np.tanh(z+1.96*se):.3f}], p = {pr:.2g}")
log("\n=== TELEPORT STEPS (all measured steps) ===")
TP = pd.read_csv(DATA / "teleports.csv"); stp = TP.step.dropna()
tele = dict(n_steps=int(len(stp)), n_teleports=int(len(TP)), median=stp.median(), q1=stp.quantile(.25), q3=stp.quantile(.75), p95=stp.quantile(.95),
            max=stp.max(), share_lt2=(stp < 2).mean(), share_lt25=(stp < 2.5).mean(), share_lt36=(stp < 3.6).mean(), share_lt05=(stp < 0.5).mean())
log.put("teleport_steps", tele)
log(f"steps measured for {tele['n_steps']} of {tele['n_teleports']} teleports: median {tele['median']:.2f} m (IQR {tele['q1']:.2f}-{tele['q3']:.2f}), "
    f"95th percentile {tele['p95']:.2f} m, max {tele['max']:.2f} m; < 2 m {tele['share_lt2']:.1%}, < 2.5 m {tele['share_lt25']:.1%}, "
    f"< 3.6 m {tele['share_lt36']:.1%}, < 0.5 m {tele['share_lt05']:.1%}")
log.save("participant_stats")
