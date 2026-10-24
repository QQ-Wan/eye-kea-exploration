"""Step 2: participant-level measures from the staged recordings.

  spatial coverage   proportion of the 100 grid cells entered (head positions outside the store dropped)
  objects looked at  unique non-structural objects with a gaze 'fixation' event (logged by the
                     experiment software after ~100 ms of gaze)
  objects available  catalogue objects (furniture, decor) in the grid cells entered, and how many of them
                     were looked at
  backtracking       2026 detector (lib/spatial_detector.py) on 5-m waypoints; duration and distance of
                     the return phase, summed over blocks; every episode is kept for later analyses
  distance traveled  path length through the 5-m waypoints, mean of the two blocks
  teleports          time of every teleport; step length for teleports with no other teleport between
                     the two surrounding head samples
  search             targets 2-10, one trial per target (the latest file if a trial was restarted);
                     success = final selection on the target; a time-out counts as a failure

Outputs (private_dir): subject_measures.csv, backtracking_episodes.csv, teleports.csv, search_trials.csv
Run steps 2 and 3 one at a time (parallel reads of cloud-synced files came back incomplete).
"""
import sys, re, logging
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from config import *

logging.disable(logging.CRITICAL)
sys.path.insert(0, str(HERE / "lib"))
from spatial_detector import SpatialBacktrackingDetector

M = pd.read_csv(PRIVATE / "manifest_exploration.csv")
staged = {(r.ID, r.block): next(STAGING.glob(f"S-{r.ID}_{r.block}_*_head-tracking.txt")) for r in M.itertuples()}
x0, x1, z0, z1 = GRID_BOUNDS

# store layout for the object-availability measures
cat = pd.read_csv(ENV / "object_catalogue.csv")
cat = cat[cat.category.isin(["furniture", "decor"])].rename(columns={"x pos": "x", "z pos": "z"})
def cell(x, z): return (min(int((x - x0) / (x1 - x0) * 10), 9), min(int((z - z0) / (z1 - z0) * 10), 9))
cat["cell"] = [cell(a, b) for a, b in zip(cat.x, cat.z)]

def head(path):
    a = [l.rstrip("\n").split("\t") for l in open(path) if l[:1].isdigit()]
    return np.array([[float(r[0]), float(r[1]), float(r[3])] for r in a])

def backtracking(path, sid):
    det = SpatialBacktrackingDetector(); det.target_distance = BT_SPACING_M
    data = det.load_navigation_data(str(path), sid)
    if data is None:
        return 0.0, 0.0, 0.0, []
    ts = data["timestamps"]; idx, sp, _ = det.distance_based_subsampling(data["positions"])
    path_len = float(np.sqrt((np.diff(sp, axis=0) ** 2).sum(1)).sum()) if len(sp) > 1 else 0.0
    dur = dist = 0.0; eps = []
    if len(sp) >= 12:
        found = det.detect_episodes(sp)
        if found:
            for e in det.filter_overlapping_episodes(found):
                dur += float(ts[idx[e["end_seq"] - 1]] - ts[idx[e["turning_point_seq"] - 1]])
                seg = sp[e["turning_point_seq"] - 1:e["end_seq"]]
                dist += float(np.sqrt((np.diff(seg, axis=0) ** 2).sum(1)).sum())
                eps.append(dict(start=float(ts[idx[e["start_seq"] - 1]]), turn=float(ts[idx[e["turning_point_seq"] - 1]]),
                                end=float(ts[idx[e["end_seq"] - 1]])))
    return dur, dist, path_len, eps

def visited(paths):
    """Grid cells entered (in-store head samples only)."""
    cells = set()
    for p in paths:
        a = head(p)
        a = a[(a[:, 1] >= x0) & (a[:, 1] <= x1) & (a[:, 2] >= z0) & (a[:, 2] <= z1)]
        if len(a):
            ix = np.minimum(((a[:, 1] - x0) / (x1 - x0) * 10).astype(int), 9)
            iz = np.minimum(((a[:, 2] - z0) / (z1 - z0) * 10).astype(int), 9)
            cells |= set(zip(ix.tolist(), iz.tolist()))
    return cells

def looked(paths):
    """Objects with a 'fixation' event."""
    fix = set()
    for p in paths:
        for l in open(p, errors="ignore"):
            parts = l.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[2].strip() == "fixation":
                fix.add(parts[1].strip())
    return fix

def nonstructural(objs): return {o for o in objs if not any(k in o.lower() for k in STRUCTURAL)}

def teleports(hpath):
    a = head(hpath)
    tp = np.array([float(l.split("\t")[0]) for l in open(str(hpath).replace("head-tracking", "interactions"), errors="ignore") if "Teleport event" in l])
    out = []
    for e in tp:
        i = np.searchsorted(a[:, 0], e); step = np.nan
        if 1 <= i < len(a) and np.sum((tp > a[i - 1, 0]) & (tp <= a[i, 0])) == 1:
            step = float(np.hypot(a[i, 1] - a[i - 1, 1], a[i, 2] - a[i - 1, 2]))
        out.append((float(e), step))
    return out

rows, ep_rows, tp_rows = [], [], []
for (sid, grp), sub in M.groupby(["ID", "AgeGroup"]):
    hp = [staged[(sid, b)] for b in ["learn1", "learn2"]]
    gp = [Path(str(p).replace("head-tracking", "gaze-events")) for p in hp]
    bt = [backtracking(p, sid) for p in hp]
    for b, res in zip(["learn1", "learn2"], bt):
        ep_rows += [dict(ID=sid, block=b, **e) for e in res[3]]
    for b, p in zip(["learn1", "learn2"], hp):
        tp_rows += [dict(ID=sid, block=b, time=t, step=s) for t, s in teleports(p)]
    steps = [r["step"] for r in tp_rows if r["ID"] == sid and not np.isnan(r["step"])]
    cells = visited(hp)
    fix = looked(gp)
    av1 = cat[cat.cell.isin(cells)]
    rows.append(dict(ID=sid, AgeGroup=grp, spatial_coverage=len(cells) / 100,
                     object_count=len(nonstructural(fix)), avail_cells=len(av1), seen_cells=int(av1.item.isin(fix).sum()),
                     BT=sum(b[0] for b in bt), BT_distance=sum(b[1] for b in bt), distance_traveled=np.mean([b[2] for b in bt]),
                     n_teleports=sum(r["ID"] == sid for r in tp_rows), step_median=np.median(steps), step_mean=np.mean(steps)))
    print(f"  {sid} done", flush=True)
D = pd.DataFrame(rows)
pd.DataFrame(ep_rows).to_csv(PRIVATE / "backtracking_episodes.csv", index=False)
pd.DataFrame(tp_rows).to_csv(PRIVATE / "teleports.csv", index=False)

# ---- search: one trial per target (latest file), targets 2-10, time-out = failure
targets = {tuple(int(v) for v in str(c).strip("\"' []").split(",")) for c in pd.read_csv(ENV / "search_targets.csv").collectable}
MON = {m: i for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1)}
def stamp_key(s):
    m = re.match(r"(\w{3})-(\d\d) (\d\d)-(\d\d)-(\d\d)", s); return (MON[m.group(1)], *map(int, m.groups()[1:]))
S = pd.read_csv(PRIVATE / "manifest_search.csv"); S = S[S.target >= 2].copy()
S["key"] = S.stamp.map(stamp_key)
S = S.sort_values("key").groupby(["ID", "target"], as_index=False).last()
def score(path):
    g = [(float(m.group(1)), tuple(int(v) for v in m.group(2).split(","))) for l in open(path, errors="ignore")
         for m in [re.search(r"^([\d.]+)\s+Collectable in grid cell \[([^\]]+)\] grabbed", l)] if m]
    return (int(g[-1][1] in targets), g[-1][0]) if g else (0, np.nan)
S[["success", "time"]] = [score(p) for p in S.path]
S[["ID", "target", "success", "time"]].to_csv(PRIVATE / "search_trials.csv", index=False)
agg = S.groupby("ID").apply(lambda x: pd.Series(dict(success_proportion=x.success.mean(), n_search_trials=len(x),
                                                     average_success_rt=x.loc[x.success == 1, "time"].mean())))
D = D.merge(agg, left_on="ID", right_index=True, how="left")
D.to_csv(PRIVATE / "subject_measures.csv", index=False)
print(f"participants: {len(D)}; backtracking episodes: {len(ep_rows)}; teleports: {len(tp_rows)}; search trials: {len(S)}")
