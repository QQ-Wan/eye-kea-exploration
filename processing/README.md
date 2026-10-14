# Processing: raw recordings → `data/`

These steps need the raw recordings, which are not shared. They are included so the derivation of every shared variable can be inspected.

## Setup

Copy `paths.example.json` to `paths.json` and set three folders:

- `raw_root`: the project folder with the recordings (`Adults/`, `Children/`, `VR_data/`, search files);
- `private_dir`: for identified intermediate files and the ID key. It must be outside the repository;
- `staging_dir`: for symbolic links with standard file names. Keep it outside cloud-synced folders; some sync services duplicate links.

`paths.json` is not committed.

## Steps

```bash
cd processing
python3 01_manifest.py      # chooses the recording for each participant x block; stages them
python3 02_measures.py      # participant-level measures, backtracking episodes, teleports, search
python3 03_events.py        # multisensory and control events (add --reprocess to rebuild the window cache)
python3 04_export.py        # recodes IDs and writes data/ and results/sample_description.txt
```

Steps 2 and 3 each read several hundred files. Run them one at a time: reading the same cloud-synced files from two processes at once gave incomplete reads.

## Files

| File | Content |
|---|---|
| `config.py` | Inclusion rule, excluded sessions with reasons, recordings saved under non-standard names, and measure settings |
| `lib/spatial_detector.py` | Backtracking detector |
| `lib/process_ms_events_simple.py` | Pre/post windows around multisensory events |
| `lib/process_control_events_simple.py` | Pre/post windows around control events |
| `environment/` | Store layout: multisensory objects (`ms_objects.csv`), matched control objects (`control_objects.csv`), object catalogue with positions (`object_catalogue.csv`), room outlines (`rooms.json`), search targets (`search_targets.csv`) |

## Recording choice

For each participant and block, the recording listed in `config.FILE_OVERRIDES` is used if there is one. Those are recordings saved under a non-standard name, each verified by recording time and duration. Otherwise the step uses the recording in the participant's data folder; if a block was restarted, the longest recording is used.
