#!/usr/bin/env bash
# Runs every analysis from data/ and writes results/.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
python3 01_participant_stats.py
Rscript 02_event_models.R
python3 03_encounters_backtracking.py
python3 04_hesitation.py
python3 05_figures.py
python3 06_model_comparison_common_sample.py
