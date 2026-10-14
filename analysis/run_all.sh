#!/usr/bin/env bash
# Runs every analysis from data/ and writes results/.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
python3 01_participant_stats.py
Rscript 02_event_models.R
python3 03_object_measures.py
python3 04_encounters_backtracking.py
python3 05_hesitation.py
