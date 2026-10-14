"""Step 3: multisensory (MS) and control events.

The modules in lib/ cut two 5-s windows around every MS event (pre: [-5, 0), post: [0, 5]) and
around control events (teleports near the object furthest from each MS object, sampled to match
each participant's number of MS events; fixed seed). This step pairs the two windows of each event
within its block and numbers repeated encounters with the same object across both blocks.

Outputs (private_dir): event_windows.csv (cache; rebuild with --reprocess), events.csv
"""
import sys, logging
sys.dont_write_bytecode = True
import pandas as pd
from config import *

logging.disable(logging.CRITICAL)
sys.path.insert(0, str(HERE / "lib"))
import process_ms_events_simple as PM
import process_control_events_simple as PC

M = pd.read_csv(PRIVATE / "manifest_exploration.csv")
group = M.drop_duplicates("ID").set_index("ID").AgeGroup

WIN = PRIVATE / "event_windows.csv"
if WIN.exists() and "--reprocess" not in sys.argv:
    raw = pd.read_csv(WIN)
else:
    op = PM.load_object_positions(str(ENV / "ms_objects.csv"))
    fo = PC.load_furthest_objects(str(ENV / "control_objects.csv"))
    ms_counts = PC.get_ms_event_counts(str(STAGING))
    rows = []
    for sid in group.index:
        ms = pd.DataFrame(PM.process_subject_all_sessions(str(STAGING), sid, op, {})); ms["event_type"] = "MS"
        n = ms_counts.get(sid, 0)
        ct = pd.DataFrame(PC.process_subject_all_sessions(str(STAGING), sid, fo, {}, {f"{sid}_learn1": n // 2, f"{sid}_learn2": n - n // 2}))
        ct["event_type"] = "Control"
        for x in (ms, ct):
            if len(x):
                x["subject_id"] = sid; x["AgeGroup"] = group[sid]; rows.append(x)
        print(f"  {sid} done", flush=True)
    raw = pd.concat(rows, ignore_index=True)
    raw.to_csv(WIN, index=False)
raw["obj"] = raw.object_name          # MS object for MS events; matched control object for control events

# ---- one row per event: post minus pre stopping time, windows paired within block
ev = []
for key, x in raw.groupby(["subject_id", "session", "event_type", "interaction_index"]):
    w0, w1 = x[x.window_index == 0], x[x.window_index == 1]
    if len(w0) != 1 or len(w1) != 1:
        continue
    f = w0.iloc[0]
    ev.append(dict(subject_id=key[0], session=key[1], event_type=key[2], AgeGroup=f.AgeGroup, obj=f.obj, t=f.window_start,
                   diff=w1.avg_stop_duration.iloc[0] - w0.avg_stop_duration.iloc[0]))
E = pd.DataFrame(ev).sort_values(["subject_id", "event_type", "obj", "session", "t"])
E["encounter"] = E.groupby(["subject_id", "event_type", "obj"]).cumcount() + 1      # counted across both blocks
E.to_csv(PRIVATE / "events.csv", index=False)
msE = E[E.event_type == "MS"]
print(f"events: {len(E)} ({len(msE)} MS, {int((E.event_type == 'Control').sum())} control) from {E.subject_id.nunique()} participants")
print(f"MS objects per participant M={msE.groupby('subject_id')['obj'].nunique().mean():.2f}; "
      f"encounters per participant-object M={msE.groupby(['subject_id', 'obj']).size().mean():.2f}")
