# Data

De-identified data derived from the VR recordings by [`processing/`](../processing/). One row per participant, event, trial, episode or teleport. `code` identifies the participant in every file: A01–A42 are adults and C01–C41 are children, assigned in random order. Times are in seconds from the start of a block and distances are in meters. Exploration blocks are `learn1` and `learn2`.

## participants.csv

| Column | Description |
|---|---|
| `group` | `adult` or `child` |
| `age` | Age in years |
| `vr_experience` | "I have a lot of experience with VR head-mounted display systems." 1–7, 7 = strongly agree |
| `game_experience` | "I have a lot of experience playing video games (e.g., phone, computer, gaming console)." 1–7, 7 = strongly agree |

The adult form ran from 1 = strongly agree to 7 = strongly disagree. Adult responses are reversed (8 − x) so that both groups use the same scale. Empty cells are missing responses.

## subject_measures.csv

| Column | Description |
|---|---|
| `spatial_coverage` | Proportion of the 100 grid cells entered across both blocks |
| `object_count` | Objects with a gaze fixation event (about 100 ms of gaze). Floors, walls, ceilings, doors and windows are excluded |
| `object_count_300_single` | Same, counting an object only if one look at it lasted > 300 ms |
| `object_count_300_total` | Same, counting an object only if total looking time exceeded 300 ms |
| `avail_cells` | Catalogued furniture and decor objects located in the grid cells the participant entered |
| `seen_cells` | Of those, objects with a fixation event. Object coverage = `seen_cells / avail_cells` |
| `seen_cells_300` | Of those, objects with one look > 300 ms |
| `avail_rooms`, `seen_rooms` | As above, for objects in the room outlines the participant entered. Objects outside every outline count as available if their grid cell was entered |
| `BT` | Backtracking duration: time on the return phase of backtracking episodes, summed over blocks |
| `BT_distance` | Path length of those return phases, summed over blocks |
| `distance_traveled` | Path length through 5 m waypoints, mean of the two blocks |
| `n_teleports` | Teleports in both blocks |
| `step_median`, `step_mean` | Median and mean teleport step length. Only teleports with no other teleport between the two surrounding head samples are used |
| `success_proportion` | Proportion of search trials ending on the target |
| `n_search_trials` | Search trials completed (out of 9) |
| `average_success_rt` | Mean search time on successful trials |

## events.csv

One row per multisensory (MS) or control event.

| Column | Description |
|---|---|
| `block` | Exploration block |
| `event_type` | `MS`: a multisensory object was triggered. `Control`: a teleport that landed within 3 m of a control object. Control objects are the objects furthest from each MS object in the same room (`processing/environment/control_objects.csv`). Control events are sampled to match each participant's number of MS events |
| `object` | MS object, or the control object |
| `window_start` | Start of the 5 s pre-event window. The event occurs at `window_start + 5` |
| `encounter` | Encounter number of this participant with this object (1 = first), counted across both blocks |
| `stop_diff` | Stopping time in the 5 s after the event minus the 5 s before. Stopping time is the mean interval between consecutive teleports in a window. With a single teleport it is the longer time to a window edge, and with no teleport it is 5 s |

## search_trials.csv

| Column | Description |
|---|---|
| `target` | Target number (2–10) |
| `success` | 1 if the object finally selected was the target, 0 otherwise (including time-outs) |
| `time` | Time of the final selection from trial start; empty if nothing was selected |

## backtracking_episodes.csv

One row per backtracking episode: `block`, `start`, `turn` and `end` (times of the first waypoint, the turning point and the last waypoint). The outbound leg runs from `start` to `turn`, and the return leg from `turn` to `end`.

## teleports.csv

One row per teleport: `block`, `time`, and `step` (distance covered). `step` is empty when another teleport fell between the same two head samples.
