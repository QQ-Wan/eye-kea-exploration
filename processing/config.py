"""Settings for the processing steps (raw recordings -> derived data).

The raw recordings are not shared. Local paths are read from processing/paths.json
(copy paths.example.json and edit it; paths.json is not committed). Every decision about
who is analyzed and which recording is used lives in this file.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ENV = HERE / "environment"                  # store layout: objects, rooms, search targets (shared)
_paths = json.load(open(HERE / "paths.json"))
RAW = Path(_paths["raw_root"])              # project folder holding the raw recordings
PRIVATE = Path(_paths["private_dir"])      # identified intermediate data and the ID key
STAGING = Path(_paths["staging_dir"])      # symlinks with standard file names, rebuilt by step 1
if PRIVATE.resolve() == REPO or REPO in PRIVATE.resolve().parents:
    raise SystemExit("private_dir must be outside the repository")
PRIVATE.mkdir(parents=True, exist_ok=True)
STAGING.mkdir(parents=True, exist_ok=True)
SHARED_DATA = REPO / "data"
SHARED_RESULTS = REPO / "results"

# ---------------------------------------------------------------- participants
# Children: session codes DVR1-DVR51. Included: both exploration blocks and the search task completed.
CHILD_EXCLUDED = {
    "DVR12": "equipment malfunction (no data recorded)",
    "DVR24": "equipment malfunction (no data recorded)",
    "DVR36": "equipment malfunction (no data recorded)",
    "DVR44": "equipment malfunction (no data recorded)",
    "DVR50": "equipment malfunction (no data recorded)",
    "DVR43": "equipment malfunction (empty exploration files)",
    "DVR45": "equipment malfunction (exploration aborted)",
    "DVR16": "first session failed; the child completed the study later as DVR49",
    "DVR42": "did not attend",
    "DVR17": "incomplete data (partial block 1, search task not completed)",
}
CHILDREN = sorted({f"DVR{i}" for i in range(1, 52)} - set(CHILD_EXCLUDED), key=lambda s: int(s[3:]))

# Adults: every participant in Adults/ (engaged, both exploration blocks).
ADULT_DIR = RAW / "Adults"

# ---------------------------------------------------------------- file sources
CHILD_DIRS = [RAW / "Children", RAW / "VR_data"]
# Blocks whose recording is saved under a non-standard name, verified by recording time and duration.
FILE_OVERRIDES = {
    ("DVR1", "learn2"): "VR_data/S-DVR1_learn2_Sep-16 15-54-13",    # full block; 15-52-09 is an 85-s restart
    ("DVR2", "learn1"): "VR_data/S-DVR2_Sep-10 12-21-51",           # saved without block labels
    ("DVR2", "learn2"): "VR_data/S-DVR2_Sep-10 12-27-12",
    ("DVR5", "learn1"): "VR_data/S-DVR5_Sep-28 10-37-44",           # saved without block labels
    ("DVR5", "learn2"): "VR_data/S-DVR5_Sep-28 10-42-56",
    ("DVR8", "learn1"): "VR_data/S-DVR8_learn1_Sep-29 11-00-15",    # 10-59-45 is an empty restart
    ("DVR8", "learn2"): "VR_data/S-DVR8_learn2_Sep-29 11-05-25",
    ("DVR9", "learn1"): "VR_data/S-DVR8_learn1_Sep-30 16-55-45",    # recorded in DVR9's session, saved under DVR8's code
    ("DVR9", "learn2"): "Children/S-DVR9_learn2_Sep-30 17-00-29",
    ("DVR33", "learn1"): "VR_data/S-DVRV2_learn1_Nov-15 18-05-36",  # recorded in DVR33's session, saved under the code DVRV2
    ("DVR33", "learn2"): "VR_data/S-DVR33_learn2_Nov-15 18-11-16",
}
FILE_TYPES = ["head-tracking", "gaze-events", "gaze-vectors", "interactions", "collectables"]

SEARCH_DIRS = [RAW / "DA/scripts/search_performance/rawdata"]
SEARCH_EXTRA = {"DVR5": RAW / "VR_data"}      # DVR5's search files

# Participant information read by the export step (identified; never shared)
AGES = RAW / "DA/scripts/2026/Regression/data/merged_complete_data_new.csv"
CHILD_DEMOGRAPHICS = RAW / "DA/scripts/demo_children.csv"
CHILD_BIRTHDATES = RAW / "DA/demographics/demo_children_raw.csv"
ADULT_DEMOGRAPHICS = RAW / "DA/scripts/roaming_beh/demographics/Demographics Questionnaire (Responses) - Form Responses 1 (1).csv"
CHILD_QUESTIONNAIRE = RAW / "DA/scripts/Mem_Questionnaire/data/children_responses.csv"
ADULT_QUESTIONNAIRE = RAW / "DA/scripts/Questionnaires/filtered_data/NAM_2_Questionnaire_clean_filtered.csv"

# ---------------------------------------------------------------- measures
GRID_BOUNDS = (-10.835, 51.456, -12.292, 56.145)   # x0, x1, z0, z1 of the store; 10 x 10 grid
STRUCTURAL = ["floor", "ceiling", "wall", "door", "window", "sky", "ground", "terrain"]
BT_SPACING_M = 5.0                                  # backtracking detector waypoint spacing
