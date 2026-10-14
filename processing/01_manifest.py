"""Step 1: decide which raw recording is used for every participant x block, and stage it.

Rule: use the entry in config.FILE_OVERRIDES if there is one; otherwise use the recording in
the first folder that has one (Adults/, or Children/ before VR_data/), taking the longest
recording if a block was restarted (restarts leave short or empty files).
Staged files are symlinks named S-<ID>_<block>_<stamp>_<type>.txt, so the processing modules
in lib/ can read them unchanged.

Outputs (private_dir): manifest_exploration.csv, manifest_search.csv
"""
import os, re
import pandas as pd
from config import *

def duration(prefix):
    p = Path(f"{prefix}_head-tracking.txt")
    if not p.exists():
        return 0.0
    t = [float(l.split("\t")[0]) for l in open(p, errors="ignore") if l[:1].isdigit()]
    return t[-1] - t[0] if len(t) > 1 else 0.0

def prefixes(folder, sid, block):
    """All recordings for one participant/block in a folder (case-insensitive ID)."""
    out = set()
    for f in os.listdir(folder):
        m = re.match(rf"(S-{re.escape(sid)}_{block}_[^_]+)_head-tracking\.txt$", f, re.I)
        if m:
            out.add(str(folder / m.group(1)))
    return out

adult_ids = sorted({re.match(r"S-([^_]+)_", f).group(1) for f in os.listdir(ADULT_DIR) if f.startswith("S-")})
rows = []
for group, ids, dirs in [("Adult", adult_ids, [ADULT_DIR]), ("Children", CHILDREN, CHILD_DIRS)]:
    for sid in ids:
        for block in ["learn1", "learn2"]:
            key = (sid.upper(), block)
            if key in FILE_OVERRIDES:
                prefix, source = str(RAW / FILE_OVERRIDES[key]), "override"
            else:
                cands = set()
                for d in dirs:                      # first folder with any recording wins
                    cands = prefixes(d, sid, block)
                    if cands:
                        break
                if not cands:
                    rows.append(dict(ID=sid.upper(), file_id=sid, AgeGroup=group, block=block, prefix=None, source="missing", duration=0.0, n_candidates=0))
                    continue
                prefix = max(cands, key=lambda p: (duration(p), "Children" in p))
                source = "longest" if len(cands) > 1 else "only"
            rows.append(dict(ID=sid.upper(), file_id=sid, AgeGroup=group, block=block, prefix=prefix,
                             source=source, duration=round(duration(prefix), 1),
                             n_candidates=1 if source == "override" else len(cands)))
M = pd.DataFrame(rows)
M.to_csv(PRIVATE / "manifest_exploration.csv", index=False)

# ---- stage symlinks with standard names
for f in STAGING.glob("*"):
    f.unlink()
for r in M.dropna(subset=["prefix"]).itertuples():
    stamp = re.search(r"([A-Z][a-z]{2}-\d\d \d\d-\d\d-\d\d)$", r.prefix).group(1)
    for t in FILE_TYPES:
        src = Path(f"{r.prefix}_{t}.txt")
        if src.exists():
            os.symlink(src, STAGING / f"S-{r.ID}_{r.block}_{stamp}_{t}.txt")

# ---- search files (all trial files; one trial per target is chosen at scoring)
srows = []
want = set(M.ID)
for d in SEARCH_DIRS:
    for f in os.listdir(d):
        m = re.match(r"S-(.+?)_target(\d+)_(.+)_interactions\.txt$", f)
        if m and m.group(1).upper() in want and m.group(1).upper() not in SEARCH_EXTRA:
            srows.append(dict(ID=m.group(1).upper(), target=int(m.group(2)), stamp=m.group(3), path=str(d / f)))
for sid, d in SEARCH_EXTRA.items():
    for f in os.listdir(d):
        m = re.match(rf"S-{sid}_target(\d+)_(.+)_interactions\.txt$", f, re.I)
        if m:
            srows.append(dict(ID=sid, target=int(m.group(1)), stamp=m.group(2), path=str(d / f)))
S = pd.DataFrame(srows)
S.to_csv(PRIVATE / "manifest_search.csv", index=False)

# ---- report
print(f"participants: adults={M[M.AgeGroup=='Adult'].ID.nunique()}, children={M[M.AgeGroup=='Children'].ID.nunique()}")
short = M[M.duration < 230]
print(f"blocks shorter than 230 s or missing: {len(short)}")
if len(short):
    print(short[["ID", "block", "source", "duration", "prefix"]].to_string(index=False))
print("blocks chosen by override:", int((M.source == "override").sum()), "| chosen as longest of several:", int((M.source == "longest").sum()))
print(f"search files: {len(S)} for {S.ID.nunique()} participants; without search files: {sorted(want - set(S.ID))}")
print(f"staged files: {len(list(STAGING.glob('*')))}")
