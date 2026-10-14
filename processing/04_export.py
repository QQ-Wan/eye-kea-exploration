"""Step 4: write the de-identified data shared in data/.

Participant IDs are replaced by codes (A01-A42 adults, C01-C41 children) assigned in random order.
The key linking codes to IDs is kept in private_dir/id_key.csv and reused on later runs, so codes
stay stable. Only age in years and two questionnaire items are shared per participant; sample
descriptives that need birthdates or demographic forms are written as group totals
(results/sample_description.txt).
Optional private corrections to demographic records: private_dir/demographic_overrides.json,
e.g. {"<ID>": {"Gender": "..."}}.
"""
import json, secrets
import numpy as np, pandas as pd
from config import *

S = pd.read_csv(PRIVATE / "subject_measures.csv")
E = pd.read_csv(PRIVATE / "events.csv")
T = pd.read_csv(PRIVATE / "search_trials.csv")
EP = pd.read_csv(PRIVATE / "backtracking_episodes.csv")
TP = pd.read_csv(PRIVATE / "teleports.csv")

# ---------------------------------------------------------------- ID key
KEY = PRIVATE / "id_key.csv"
if KEY.exists():
    key = pd.read_csv(KEY)
else:
    rows = []
    for grp, letter in [("Adult", "A"), ("Children", "C")]:
        ids = sorted(S[S.AgeGroup == grp].ID)
        order = sorted(ids, key=lambda _: secrets.randbits(64))
        rows += [dict(ID=i, code=f"{letter}{n:02d}") for n, i in enumerate(order, 1)]
    key = pd.DataFrame(rows); key.to_csv(KEY, index=False)
code = key.set_index("ID").code
missing = sorted(set(S.ID) - set(code.index))
if missing:
    raise SystemExit(f"participants without a code: {missing}")

# ---------------------------------------------------------------- participant information
overrides = json.load(open(PRIVATE / "demographic_overrides.json")) if (PRIVATE / "demographic_overrides.json").exists() else {}
P = pd.read_csv(AGES); P["ID"] = P.SUBJECT_ID.str.upper()
age = P.drop_duplicates("ID").set_index("ID").Age.to_dict()
demo = pd.read_csv(CHILD_DEMOGRAPHICS); demo["ID"] = demo.subject_id.str.upper().str.strip(); demo = demo.drop_duplicates("ID")
age.update({i: a for i, a in zip(demo.ID, demo.Age) if i not in age})
S["Age"] = S.ID.map(age)
if S.Age.isna().any():
    raise SystemExit(f"missing age: {sorted(S[S.Age.isna()].ID)}")

items = {"vr_experience": "I have a lot of experience with VR head-mounted display systems.",
         "game_experience": "I have a lot of experience playing video games (e.g., phone, computer, gaming console)."}
cq = pd.read_csv(CHILD_QUESTIONNAIRE); cq.columns = [c.strip() for c in cq.columns]
cq["ID"] = cq["Enter your Subject ID."].astype(str).str.upper().str.strip(); cq = cq.drop_duplicates("ID", keep="last").set_index("ID")
aq = pd.read_csv(ADULT_QUESTIONNAIRE); aq.columns = [c.strip() for c in aq.columns]
aq["ID"] = aq.analysis_subject_id.astype(str).str.upper(); aq = aq.drop_duplicates("ID", keep="last").set_index("ID")
part = S[["ID", "AgeGroup", "Age"]].copy()
for name, item in items.items():
    child = pd.to_numeric(cq[item], errors="coerce")             # child form: 1 = strongly disagree, 7 = strongly agree
    adult = 8 - pd.to_numeric(aq[item], errors="coerce")         # adult form: 1 = strongly agree, 7 = strongly disagree
    part[name] = [child.get(i) if g == "Children" else adult.get(i) for i, g in zip(part.ID, part.AgeGroup)]

# ---------------------------------------------------------------- shared files
def out(df, name, cols, sort):
    df = df.assign(code=df[[c for c in ("ID", "subject_id") if c in df][0]].map(code))
    df = df.rename(columns={"AgeGroup": "group"})
    if "group" in df:
        df["group"] = df.group.map({"Adult": "adult", "Children": "child"})
    df[["code"] + cols].sort_values(["code"] + sort).to_csv(SHARED_DATA / name, index=False)

out(part.rename(columns={"Age": "age"}), "participants.csv", ["group", "age", "vr_experience", "game_experience"], [])
out(S, "subject_measures.csv",
    ["group", "spatial_coverage", "object_count", "object_count_300_single", "object_count_300_total",
     "avail_cells", "seen_cells", "seen_cells_300", "avail_rooms", "seen_rooms",
     "BT", "BT_distance", "distance_traveled", "n_teleports", "step_median", "step_mean",
     "success_proportion", "n_search_trials", "average_success_rt"], [])
out(E.rename(columns={"session": "block", "obj": "object", "t": "window_start", "diff": "stop_diff"}), "events.csv",
    ["group", "block", "event_type", "object", "window_start", "encounter", "stop_diff"], ["event_type", "block", "window_start"])
out(T, "search_trials.csv", ["target", "success", "time"], ["target"])
out(EP, "backtracking_episodes.csv", ["block", "start", "turn", "end"], ["block", "start"])
out(TP, "teleports.csv", ["block", "time", "step"], ["block", "time"])

# ---------------------------------------------------------------- sample description (group totals only)
lines = []
A, C = S[S.AgeGroup == "Adult"], S[S.AgeGroup == "Children"]
lines.append(f"adults n = {len(A)}; children n = {len(C)}")
lines.append(f"children by year of age: {C.Age.astype(int).value_counts().sort_index().to_dict()}")
raw = pd.read_csv(CHILD_BIRTHDATES, sep="\t"); raw.columns = [c.strip() for c in raw.columns]
raw["ID"] = raw.id.astype(str).str.upper().str.strip()
raw = raw[raw.ID.isin(C.ID)].dropna(subset=["birthdate", "task_date"])
dec = ((pd.to_datetime(raw.task_date, errors="coerce") - pd.to_datetime(raw.birthdate, errors="coerce")).dt.days / 365.25).dropna()
lines.append(f"children, decimal age (n = {len(dec)} with birthdates): M = {dec.mean():.2f}, SD = {dec.std():.2f}, range {dec.min():.2f}-{dec.max():.2f}")
g = demo.set_index("ID").Gender.reindex(C.ID).copy()
for i, o in overrides.items():
    if "Gender" in o and i in g.index:
        g[i] = o["Gender"]
lines.append(f"children, gender: {g.value_counts().to_dict()}")
ad = pd.read_csv(ADULT_DEMOGRAPHICS); ad["ID"] = ad["Enter your Subject ID: "].astype(str).str.upper().str.strip()
ad = ad[ad.ID.isin(A.ID)].drop_duplicates("ID", keep="last")
lines.append(f"adults, age: M = {A.Age.mean():.2f}, SD = {A.Age.std():.2f}, range {A.Age.min():.0f}-{A.Age.max():.0f}")
lines.append(f"adults, gender (n = {len(ad)} forms matched): {ad['Enter your gender identity:'].str.strip().str.title().value_counts().to_dict()}")
(SHARED_RESULTS / "sample_description.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"shared files written to {SHARED_DATA}")
