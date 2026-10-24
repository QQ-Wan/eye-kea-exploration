"""Analysis settings. Every analysis runs from the shared, de-identified data in data/."""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)

OUTLIER_RULE = "both"   # exclude from a regression model only if Cook's D > 4/n AND |studentized residual| > 2.5
MAX_ENCOUNTER = 4       # first four encounters with each multisensory object, counted across both blocks
