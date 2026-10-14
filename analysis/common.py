"""Data loading and statistical helpers shared by the Python analysis scripts.

Conventions: group differences are adults minus children; Cohen's d uses the pooled SD;
regressions are OLS with adults as the reference group; the outlier rule is applied
separately to each regression model, and results without exclusions are kept alongside.
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
    D["opt1"] = D.seen_cells / D.avail_cells            # Option 1: share of objects in the grid cells entered that were looked at
    D["opt1_300"] = D.seen_cells_300 / D.avail_cells    #   same, looked at for > 300 ms in one look
    D["opt2"] = D.seen_rooms / D.avail_rooms            # Option 2: share of objects in the rooms entered that were looked at
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

def partial_r(x, a="BT", b="adaptation", covs="age + n_ms_events"):
    """Partial correlation of a and b controlling for covs, with a Fisher-z 95% CI."""
    ra = smf.ols(f"{a} ~ {covs}", x).fit().resid; rb = smf.ols(f"{b} ~ {covs}", x).fit().resid
    r, p = st.pearsonr(ra, rb); k = len(covs.split("+")); z = np.arctanh(r); se = 1 / np.sqrt(len(x) - 3 - k)
    return dict(r=r, df=len(x) - 2 - k, p=p, CI=[np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)], n=len(x))

# joint models compared for each outcome (numbered 2-19 as in Tables S1-S3); predictors centered
TERMS = {"B": "BT_c", "A": "adaptation_c", "G": "group"}
MODELS = [["B"], ["A"], ["G"], ["B", "A"], ["B", "G"], ["A", "G"], ["B", "A", "G"], ["B", "A", "B:A"], ["B", "G", "B:G"], ["A", "G", "A:G"],
          ["B", "A", "G", "B:A"], ["B", "A", "G", "B:G"], ["B", "A", "G", "A:G"], ["B", "A", "G", "B:A", "B:G"], ["B", "A", "G", "B:A", "A:G"],
          ["B", "A", "G", "B:G", "A:G"], ["B", "A", "G", "B:A", "B:G", "A:G"], ["B", "A", "G", "B:A", "B:G", "A:G", "B:A:G"]]
def rhs(spec): return " + ".join(":".join(TERMS[t] for t in s.split(":")) for s in spec)

def select_model(D, y, extra=""):
    """Fit models 2-19 on participants with all predictors; choose by the majority of AIC, BIC and adjusted R2
    (AIC if all three disagree). Returns the comparison table, the three picks and the chosen model number."""
    X = D.dropna(subset=[y, "BT", "adaptation"]).reset_index(drop=True)
    T = pd.DataFrame([dict(model=i, k=int(m.df_model) + 1, AIC=m.aic, BIC=m.bic, adjR2=m.rsquared_adj, formula=rhs(spec) + extra)
                      for i, spec in enumerate(MODELS, start=2) for m in [smf.ols(f"{y} ~ {rhs(spec)}{extra}", X).fit()]])
    picks = {"AIC": int(T.loc[T.AIC.idxmin(), "model"]), "BIC": int(T.loc[T.BIC.idxmin(), "model"]), "adjR2": int(T.loc[T.adjR2.idxmax(), "model"])}
    vc = pd.Series(picks).value_counts()
    return T, picks, int(vc.index[0]) if vc.iloc[0] >= 2 else picks["AIC"]

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
