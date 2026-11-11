"""Data loading and statistical helpers shared by the Python analysis scripts.

Conventions: group differences are adults minus children; Cohen's d uses the pooled SD;
regressions are OLS with adults as the reference group. Outlier rule: Cook's D > 4/n AND
|externally studentized residual| > 2.5. For the models relating backtracking and adaptation to an outcome,
it is applied once per outcome, in the full model (common_sample); age regressions apply it to each model
(fit_rule, reg_rule). Results without exclusions are kept alongside.
"""
import numpy as np, pandas as pd, scipy.stats as st, statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence
from config import *

CH = "[T.child]"

def adaptation_slopes(events, max_encounter=MAX_ENCOUNTER):
    """Per participant: OLS slope of post- minus pre-event stopping time on encounter number (MS events)."""
    ms = events[(events.event_type == "MS") & (events.encounter <= max_encounter)]
    sl = {c: np.polyfit(g.encounter, g.stop_diff, 1)[0] for c, g in ms.groupby("code") if g.encounter.nunique() >= 2}
    return pd.DataFrame({"adaptation": pd.Series(sl, dtype=float), "n_ms_events": ms.groupby("code").size()}).rename_axis("code").reset_index()

def load(max_encounter=MAX_ENCOUNTER):
    """Participant-level table (measures, age, questionnaire items, adaptation slope) and the event table."""
    P = pd.read_csv(DATA / "participants.csv")
    S = pd.read_csv(DATA / "subject_measures.csv")
    E = pd.read_csv(DATA / "events.csv")
    D = S.merge(P[["code", "age", "vr_experience", "game_experience"]], on="code").merge(adaptation_slopes(E, max_encounter), on="code", how="left")
    D["opt1"] = D.seen_cells / D.avail_cells            # object coverage: share of objects in the grid cells entered that were looked at
    D["group"] = pd.Categorical(D.group, ["adult", "child"])
    for v in ["BT", "adaptation"]:
        D[v + "_c"] = D[v] - D[v].mean()
    return D, E

def ttest(D, var, welch=False):
    a, c = D.loc[D.group == "adult", var].dropna(), D.loc[D.group == "child", var].dropna()
    t, p = st.ttest_ind(a, c, equal_var=not welch)
    va, vc = a.var(ddof=1) / len(a), c.var(ddof=1) / len(c)
    df = (va + vc) ** 2 / (va ** 2 / (len(a) - 1) + vc ** 2 / (len(c) - 1)) if welch else len(a) + len(c) - 2
    se = np.sqrt(va + vc) if welch else np.sqrt(((len(a) - 1) * a.var() + (len(c) - 1) * c.var()) / (len(a) + len(c) - 2) * (1 / len(a) + 1 / len(c)))
    md = a.mean() - c.mean(); q = st.t.ppf(.975, df)
    sp = np.sqrt(((len(a) - 1) * a.var() + (len(c) - 1) * c.var()) / (len(a) + len(c) - 2))
    return dict(adult_M=a.mean(), adult_SD=a.std(), child_M=c.mean(), child_SD=c.std(), n_adult=len(a), n_child=len(c),
                t=t, df=df, p=p, Mdiff=md, CI=[md - q * se, md + q * se], d=md / sp)

def reg(df, formula, term):
    m = smf.ols(formula, df).fit(); ci = m.conf_int().loc[term]
    return dict(b=m.params[term], CI=[ci[0], ci[1]], t=m.tvalues[term], df=m.df_resid, p=m.pvalues[term], n=int(m.nobs))

def lin(m, terms):
    """Estimate, CI and test of the sum of the named coefficients (e.g. the children's simple slope)."""
    w = pd.Series(0.0, index=m.params.index); w[terms] = 1
    r = m.t_test(w.values); ci = np.squeeze(r.conf_int())
    return dict(b=float(np.squeeze(r.effect)), CI=[float(ci[0]), float(ci[1])], t=float(np.squeeze(r.tvalue)), df=m.df_resid, p=float(np.squeeze(r.pvalue)))

def fit_rule(df, formula, need):
    """Fit, apply the outlier rule, refit. Returns (model after exclusions, model on all data, excluded codes)."""
    x = df.dropna(subset=need).reset_index(drop=True)
    m = smf.ols(formula, x).fit(); inf = OLSInfluence(m)
    ck, rs = inf.cooks_distance[0], np.abs(inf.resid_studentized_external)
    mask = (ck > 4 / len(x)) & (rs > 2.5) if OUTLIER_RULE == "both" else (ck > 4 / len(x)) | (rs > 2.5)
    return smf.ols(formula, x[~mask]).fit(), m, x.code[mask].tolist()

def reg_rule(df, formula, term):
    """reg() with the outlier rule (as in fit_rule); the all-data estimate and the excluded codes are kept alongside."""
    y = formula.split("~")[0].strip()
    m, m_all, ex = fit_rule(df, formula, [y, term])
    def coef(mm):
        ci = mm.conf_int().loc[term]
        return dict(b=mm.params[term], CI=[ci[0], ci[1]], t=mm.tvalues[term], df=mm.df_resid, p=mm.pvalues[term], n=int(mm.nobs))
    return dict(coef(m), excluded=ex, all_data=coef(m_all))

def partial_r(x, a="BT", b="adaptation", covs="age + n_ms_events"):
    """Partial correlation of a and b controlling for covs, with a Fisher-z 95% CI."""
    ra = smf.ols(f"{a} ~ {covs}", x).fit().resid; rb = smf.ols(f"{b} ~ {covs}", x).fit().resid
    r, p = st.pearsonr(ra, rb); k = len(covs.split("+")); z = np.arctanh(r); se = 1 / np.sqrt(len(x) - 3 - k)
    return dict(r=r, df=len(x) - 2 - k, p=p, CI=[np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)], n=len(x))

# models compared for each outcome (Tables S2-S4): model 1 = intercept only; models 2-19 = the 18 combinations of backtracking,
# adaptation, age group and their interactions; predictors centered at the mean of the common sample
TERMS = {"B": "BT_c", "A": "adaptation_c", "G": "group"}
MODELS = [["B"], ["A"], ["G"], ["B", "A"], ["B", "G"], ["A", "G"], ["B", "A", "G"], ["B", "A", "B:A"], ["B", "G", "B:G"], ["A", "G", "A:G"],
          ["B", "A", "G", "B:A"], ["B", "A", "G", "B:G"], ["B", "A", "G", "A:G"], ["B", "A", "G", "B:A", "B:G"], ["B", "A", "G", "B:A", "A:G"],
          ["B", "A", "G", "B:G", "A:G"], ["B", "A", "G", "B:A", "B:G", "A:G"], ["B", "A", "G", "B:A", "B:G", "A:G", "B:A:G"]]
SPECS = [[]] + MODELS                                   # index i - 1 = model i
def rhs(spec): return " + ".join(":".join(TERMS[t] for t in s.split(":")) for s in spec)
def model_formula(y, i): return f"{y} ~ {rhs(SPECS[i - 1])}" if i > 1 else f"{y} ~ 1"

def center(X):
    X = X.copy()
    for v in ["BT", "adaptation"]: X[v + "_c"] = X[v] - X[v].mean()
    return X

def common_sample(D, y):
    """One sample per outcome: fit the full model (model 19) to all participants with the outcome and both predictors, exclude the
    participants the outlier rule flags, and center backtracking and adaptation at the mean of the remaining sample.
    Returns (common sample, all participants, excluded codes); the all-participant table is centered at its own mean."""
    X_all = center(D.dropna(subset=[y, "BT", "adaptation"]).reset_index(drop=True))
    inf = OLSInfluence(smf.ols(model_formula(y, 19), X_all).fit())
    flag = (inf.cooks_distance[0] > 4 / len(X_all)) & (np.abs(inf.resid_studentized_external) > 2.5)
    return center(X_all[~flag].reset_index(drop=True)), X_all, sorted(X_all.code[flag])

def select_aic_bic(T):
    """The model favored by both AIC and BIC; when they disagree, the model better by more than 2 on one criterion and within 2 on
    the other (otherwise undetermined, chosen = None). T has columns model, AIC, BIC."""
    T = T.set_index("model"); a, b = int(T.AIC.idxmin()), int(T.BIC.idxmin())
    if a == b: return dict(AIC_pick=a, BIC_pick=b, chosen=a, how="AIC and BIC agree")
    dA, dB = T.AIC[b] - T.AIC[a], T.BIC[a] - T.BIC[b]   # how much better each pick is on its own criterion
    r = dict(AIC_pick=a, BIC_pick=b, dAIC=dA, dBIC=dB)
    if dA > 2 and dB <= 2: return dict(r, chosen=a, how=f"AIC pick better by {dA:.2f} on AIC and within 2 on BIC ({dB:.2f})")
    if dB > 2 and dA <= 2: return dict(r, chosen=b, how=f"BIC pick better by {dB:.2f} on BIC and within 2 on AIC ({dA:.2f})")
    return dict(r, chosen=None, how=f"undetermined: AIC difference {dA:.2f}, BIC difference {dB:.2f}")

def fmt(r):
    return f"b={r['b']:+.4f} [{r['CI'][0]:+.4f}, {r['CI'][1]:+.4f}], t({r['df']:.0f})={r['t']:+.2f}, p={r['p']:.4f}"

class Log:
    def __init__(self): self.lines, self.out = [], {}
    def __call__(self, s=""): self.lines.append(s); print(s)
    def put(self, k, v): self.out[k] = v
    def save(self, stem):
        import json
        json.dump(self.out, open(RESULTS / f"{stem}.json", "w"), indent=1, default=float)
        (RESULTS / f"{stem}.txt").write_text("\n".join(self.lines) + "\n")
