"""Settings for the processing steps (raw recordings -> derived data).

The raw recordings are not shared. Local paths are read from processing/paths.json (copy paths.example.json and edit it;
paths.json is not committed). Settings that identify individual sessions (raw session codes, which sessions were excluded and why,
recordings saved under non-standard file names, and the locations of the identified participant records) are read from
processing_settings.json in the private folder, which is never shared. The rules they implement are stated below.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ENV = HERE / "environment"                  # store layout: objects, rooms, search targets (shared)
_paths = json.load(open(HERE / "paths.json"))
RAW = Path(_paths["raw_root"])              # project folder holding the raw recordings
PRIVATE = Path(_paths["private_dir"])      # identified intermediate data, the ID key and processing_settings.json
STAGING = Path(_paths["staging_dir"])      # symlinks with standard file names, rebuilt by step 1
if PRIVATE.resolve() == REPO or REPO in PRIVATE.resolve().parents:
    raise SystemExit("private_dir must be outside the repository")
PRIVATE.mkdir(parents=True, exist_ok=True)
STAGING.mkdir(parents=True, exist_ok=True)
SHARED_DATA = REPO / "data"
SHARED_RESULTS = REPO / "results"
_private = json.load(open(PRIVATE / "processing_settings.json"))

# ---------------------------------------------------------------- participants
# Children are included if they completed both exploration blocks and the search task. Sessions are excluded for equipment
# malfunction (no data, empty files or an aborted block), incomplete data, non-attendance, or a repeat session by the same child
# (the completed session is kept). The session list and the reason for each exclusion are in the private settings.
CHILD_EXCLUDED = _private["child_excluded"]
CHILDREN = [s for s in _private["child_sessions"] if s not in CHILD_EXCLUDED]

# Adults: every participant in Adults/ (engaged, both exploration blocks).
ADULT_DIR = RAW / "Adults"

# ---------------------------------------------------------------- file sources
CHILD_DIRS = [RAW / "Children", RAW / "VR_data"]
# Blocks whose recording was saved under a non-standard file name (for example without block labels, or after a restart), each
# verified by recording time and duration. The list of (session, block) -> recording is in the private settings.
FILE_OVERRIDES = {(s, b): path for s, b, path in _private["file_overrides"]}
FILE_TYPES = ["head-tracking", "gaze-events", "gaze-vectors", "interactions", "collectables"]

SEARCH_DIRS = [RAW / p for p in _private["search_dirs"]]
SEARCH_EXTRA = {s: RAW / p for s, p in _private["search_extra"].items()}   # search files stored with a session's recordings

# Participant information read by the export step (identified; never shared)
_files = {k: RAW / p for k, p in _private["participant_files"].items()}
AGES, CHILD_DEMOGRAPHICS, CHILD_BIRTHDATES = _files["AGES"], _files["CHILD_DEMOGRAPHICS"], _files["CHILD_BIRTHDATES"]
ADULT_DEMOGRAPHICS, CHILD_QUESTIONNAIRE, ADULT_QUESTIONNAIRE = _files["ADULT_DEMOGRAPHICS"], _files["CHILD_QUESTIONNAIRE"], _files["ADULT_QUESTIONNAIRE"]

# ---------------------------------------------------------------- measures
GRID_BOUNDS = (-10.835, 51.456, -12.292, 56.145)   # x0, x1, z0, z1 of the store; 10 x 10 grid
STRUCTURAL = ["floor", "ceiling", "wall", "door", "window", "sky", "ground", "terrain"]
BT_SPACING_M = 5.0                                  # backtracking detector waypoint spacing
