#!/usr/bin/env python3
"""
Process MS events with simplified 2-window structure.
Creates exactly 2 windows per MS event: [-5,0) and [0,5] around "participants triggered distraction" events.
"""

import pandas as pd
import numpy as np
import os
import glob
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_subject_id_from_filename(filepath: str) -> str:
    """Extract standardized subject ID from filename.

    Args:
        filepath: Path like 'S-DVR0_Learn1_Jan-01 10-00-00_interactions.txt'

    Returns:
        Standardized subject ID like 'DVR0'
    """
    filename = os.path.basename(filepath)
    # Match pattern S-{subject_id}_{session}_
    match = re.match(r'S-([^_]+)_', filename)
    if match:
        subject_id = match.group(1).upper()
        return subject_id
    else:
        logging.warning(f"Could not extract subject ID from filename: {filename}")
        return "UNKNOWN"

def extract_original_subject_id_from_filename(filepath: str) -> str:
    """Extract original case subject ID from filename for file pattern matching.

    Args:
        filepath: Path like 'S-AB123_learn1_Jan-01 10-00-00_interactions.txt'

    Returns:
        Original case subject ID like 'AB123'
    """
    filename = os.path.basename(filepath)
    # Match pattern S-{subject_id}_{session}_
    match = re.match(r'S-([^_]+)_', filename)
    if match:
        subject_id = match.group(1)  # Keep original case
        return subject_id
    else:
        logging.warning(f"Could not extract subject ID from filename: {filename}")
        return "UNKNOWN"

def determine_age_group_from_path(filepath: str) -> str:
    """Determine age group from file path.

    Args:
        filepath: Full path to the data file

    Returns:
        'Children' or 'Adult'
    """
    if '/Children/' in filepath:
        return 'Children'
    elif '/Adults/' in filepath or '/test_data/' in filepath:
        return 'Adult'  # Assume test_data is adult data
    else:
        logging.warning(f"Could not determine age group from path: {filepath}")
        return 'Adult'  # Default to Adult

def read_head_tracking_file(filepath: str) -> pd.DataFrame:
    """Read head tracking data from file."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Find the start of data (skip headers)
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip() and not any(header in line for header in ['Date and time', 'SubjectID', 'seconds', 'pos x', 'yaw']):
                data_start = i
                break

        # Read the data
        data = []
        for line in lines[data_start:]:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 7:  # Need at least 7 columns
                    try:
                        data.append({
                            'seconds': float(parts[0]),
                            'pos x': float(parts[1]),
                            'pos y': float(parts[2]),
                            'pos z': float(parts[3]),
                            'yaw': float(parts[4]),
                            'pitch': float(parts[5]),
                            'roll': float(parts[6])
                        })
                    except ValueError:
                        continue

        df = pd.DataFrame(data)
        return df
    except Exception as e:
        logging.error(f"Error reading head tracking file {filepath}: {e}")
        return pd.DataFrame()

def read_gaze_vectors_file(filepath: str) -> pd.DataFrame:
    """Read gaze vectors data from file."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Find the start of data (skip headers)
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip() and not any(header in line for header in ['Date and time', 'SubjectID', 'seconds', 'local x', 'world x']):
                data_start = i
                break

        # Read the data
        data = []
        for line in lines[data_start:]:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 7:  # Need at least 7 columns
                    try:
                        data.append({
                            'seconds': float(parts[0]),
                            'local_x': float(parts[1]),
                            'local_y': float(parts[2]),
                            'local_z': float(parts[3]),
                            'world_x': float(parts[4]),
                            'world_y': float(parts[5]),
                            'world_z': float(parts[6])
                        })
                    except ValueError:
                        continue

        df = pd.DataFrame(data)
        return df
    except Exception as e:
        logging.error(f"Error reading gaze vectors file {filepath}: {e}")
        return pd.DataFrame()

def read_gaze_events_file(filepath: str) -> pd.DataFrame:
    """Read gaze events data from file."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Find the start of data (skip headers and empty lines)
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip() and not any(header in line for header in ['Date and time', 'SubjectID', 'seconds', 'item', 'event']):
                data_start = i
                break

        # Parse fixation events to reconstruct fixation durations
        fixation_events = []
        current_fixations = {}  # Track ongoing fixations by object

        for line in lines[data_start:]:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    try:
                        time = float(parts[0])
                        obj = parts[1].strip()
                        event_type = parts[2].strip()

                        if event_type == 'start':
                            # Start of a potential fixation
                            current_fixations[obj] = time
                        elif event_type == 'end':
                            # End of fixation - calculate duration
                            if obj in current_fixations:
                                start_time = current_fixations[obj]
                                duration = time - start_time
                                if len(parts) >= 4:
                                    # Sometimes duration is provided in 4th column
                                    try:
                                        provided_duration = float(parts[3])
                                        duration = provided_duration
                                    except:
                                        pass

                                fixation_events.append({
                                    'start_time': start_time,
                                    'end_time': time,
                                    'duration': duration,
                                    'object': obj
                                })
                                del current_fixations[obj]
                    except ValueError:
                        # Skip lines that can't be parsed as data
                        continue

        return pd.DataFrame(fixation_events)
    except Exception as e:
        logging.error(f"Error reading gaze events file {filepath}: {e}")
        return pd.DataFrame()

def read_interactions_file(filepath: str) -> Tuple[List[Dict], List[Dict]]:
    """Read interaction events from file and extract MS events and all interactions."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()[4:]  # Skip header lines

        ms_events = []
        all_interactions = []

        for line in lines:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    time = float(parts[0])
                    interaction = parts[1].strip()

                    # Add all interactions for teleport propensity calculation
                    all_interactions.append({
                        'time': time,
                        'interaction': interaction
                    })

                    # Look for "participant triggered distraction {object}"
                    if interaction.startswith('participant triggered distraction '):
                        object_name = interaction.replace('participant triggered distraction ', '').strip()
                        # Skip robot events as specified
                        if object_name != 'robot':
                            ms_events.append({
                                'time': time,
                                'object_name': object_name
                            })

        return ms_events, all_interactions
    except Exception as e:
        logging.error(f"Error reading interactions file {filepath}: {e}")
        return [], []

def load_object_positions(distractors_file: str) -> Dict[str, Tuple[float, float]]:
    """Load object positions from distractors file."""
    try:
        df = pd.read_csv(distractors_file)
        positions = {}
        for _, row in df.iterrows():
            positions[row['output_name']] = (row['posx'], row['posz'])
        return positions
    except Exception as e:
        logging.error(f"Error loading object positions: {e}")
        return {}

def load_age_info(age_file: str) -> Dict[str, float]:
    """Load age information."""
    try:
        df = pd.read_csv(age_file)
        age_dict = {}
        for _, row in df.iterrows():
            age_dict[str(row['subject_id']).upper()] = row['Age']
        return age_dict
    except Exception as e:
        logging.error(f"Error loading age info: {e}")
        return {}

def calculate_distance_metrics(head_data: pd.DataFrame, object_pos: Tuple[float, float],
                              window_start: float, window_end: float) -> Dict:
    """Calculate distance metrics for a time window."""
    # Filter head data for this window
    window_data = head_data[
        (head_data['seconds'] >= window_start) &
        (head_data['seconds'] < window_end)
    ].copy()

    if len(window_data) == 0:
        return {
            'mean_distance': np.nan,
            'distance_change': np.nan,
            'min_distance': np.nan,
            'max_distance': np.nan,
            'distance_variance': np.nan,
            'initial_distance': np.nan,
            'final_distance': np.nan,
            'travelled_distance': np.nan,
            'relative_distance_change': np.nan
        }

    # Calculate distances to object
    obj_x, obj_z = object_pos
    distances = np.sqrt((window_data['pos x'] - obj_x)**2 + (window_data['pos z'] - obj_z)**2)

    # Calculate metrics
    mean_distance = np.mean(distances)
    min_distance = np.min(distances)
    max_distance = np.max(distances)
    distance_variance = np.var(distances)
    initial_distance = distances.iloc[0]
    final_distance = distances.iloc[-1]
    distance_change = final_distance - initial_distance

    # Calculate travelled distance (total path length)
    head_positions = window_data[['pos x', 'pos z']].values
    travelled_distance = 0
    if len(head_positions) > 1:
        for i in range(1, len(head_positions)):
            travelled_distance += np.sqrt(
                (head_positions[i][0] - head_positions[i-1][0])**2 +
                (head_positions[i][1] - head_positions[i-1][1])**2
            )

    # Relative distance change as proportion
    relative_distance_change = distance_change / initial_distance if initial_distance != 0 else 0

    return {
        'mean_distance': mean_distance,
        'distance_change': distance_change,
        'min_distance': min_distance,
        'max_distance': max_distance,
        'distance_variance': distance_variance,
        'initial_distance': initial_distance,
        'final_distance': final_distance,
        'travelled_distance': travelled_distance,
        'relative_distance_change': relative_distance_change
    }

def calculate_head_metrics(head_data: pd.DataFrame, window_start: float, window_end: float) -> Dict:
    """Calculate head movement metrics for a time window."""
    window_data = head_data[
        (head_data['seconds'] >= window_start) &
        (head_data['seconds'] < window_end)
    ].copy()

    if len(window_data) == 0:
        return {
            'head_samples': 0,
            'tap': np.nan,
            'mean_angular_velocity': np.nan,
            'peak_angular_velocity': np.nan,
            'angular_velocity_variance': np.nan,
            'tap_rate': np.nan,
            'movement_frequency': np.nan,
            'stillness_duration': np.nan
        }

    head_samples = len(window_data)
    window_duration = window_end - window_start

    # Calculate total angular path (TAP) and angular velocities
    tap = 0
    angular_velocities = []
    movement_threshold = 3.0  # degrees per second
    stillness_time = 0
    movement_count = 0

    if head_samples > 1:
        # Calculate angular differences and velocities between consecutive samples
        for i in range(1, len(window_data)):
            # Calculate angular difference using yaw, pitch, roll
            prev_rot = window_data.iloc[i-1][['yaw', 'pitch', 'roll']].values
            curr_rot = window_data.iloc[i][['yaw', 'pitch', 'roll']].values
            angular_diff = np.sqrt(np.sum((curr_rot - prev_rot)**2))
            tap += angular_diff

            # Calculate time interval and angular velocity
            time_diff = window_data.iloc[i]['seconds'] - window_data.iloc[i-1]['seconds']
            if time_diff > 0:
                angular_velocity = angular_diff / time_diff
                angular_velocities.append(angular_velocity)

                # Check for stillness and movement events
                if angular_velocity < movement_threshold:
                    stillness_time += time_diff
                else:
                    movement_count += 1

    # Calculate angular velocity metrics
    if len(angular_velocities) > 0:
        mean_angular_velocity = np.mean(angular_velocities)
        peak_angular_velocity = np.max(angular_velocities)
        angular_velocity_variance = np.var(angular_velocities)
    else:
        mean_angular_velocity = 0.0
        peak_angular_velocity = 0.0
        angular_velocity_variance = 0.0

    # Calculate rate-normalized metrics
    tap_rate = tap / window_duration if window_duration > 0 else 0.0
    movement_frequency = movement_count / window_duration if window_duration > 0 else 0.0
    stillness_duration = stillness_time

    return {
        'head_samples': head_samples,
        'tap': tap,
        'mean_angular_velocity': mean_angular_velocity,
        'peak_angular_velocity': peak_angular_velocity,
        'angular_velocity_variance': angular_velocity_variance,
        'tap_rate': tap_rate,
        'movement_frequency': movement_frequency,
        'stillness_duration': stillness_duration
    }

def calculate_gaze_metrics(gaze_vectors_data: pd.DataFrame, gaze_events_data: pd.DataFrame,
                          object_name: str, window_start: float, window_end: float, subject_id: str = "") -> Dict:
    """Calculate gaze metrics for a time window."""

    # Calculate fixation spread using gaze vectors (spatial dispersion)
    fixation_spread = 0.0
    window_vectors = gaze_vectors_data[
        (gaze_vectors_data['seconds'] >= window_start) &
        (gaze_vectors_data['seconds'] < window_end)
    ].copy()

    if len(window_vectors) > 1:
        # Calculate spatial spread of gaze directions using world coordinates
        x_coords = window_vectors['world_x']
        y_coords = window_vectors['world_y']
        z_coords = window_vectors['world_z']

        # Calculate 3D spatial spread
        x_std = np.std(x_coords)
        y_std = np.std(y_coords)
        z_std = np.std(z_coords)

        fixation_spread = np.sqrt(x_std**2 + y_std**2 + z_std**2)

    # Calculate object looking duration using gaze events
    object_looking_duration = 0.0

    # Check if gaze_events_data is empty or missing required columns
    if len(gaze_events_data) == 0:
        # This is normal - not all objects are looked at during every time window
        return {
            'fixation_spread': fixation_spread,
            'object_looking_duration': object_looking_duration
        }

    if 'start_time' not in gaze_events_data.columns:
        logging.warning(f"Gaze events data missing required columns for {subject_id}")
        return {
            'fixation_spread': fixation_spread,
            'object_looking_duration': object_looking_duration
        }

    window_gaze_events = gaze_events_data[
        (gaze_events_data['start_time'] < window_end) &
        (gaze_events_data['end_time'] > window_start)
    ].copy()

    if len(window_gaze_events) > 0:
        object_gaze = window_gaze_events[window_gaze_events['object'] == object_name]

        for _, fixation in object_gaze.iterrows():
            # Calculate overlap with window
            overlap_start = max(fixation['start_time'], window_start)
            overlap_end = min(fixation['end_time'], window_end)
            if overlap_end > overlap_start:
                object_looking_duration += overlap_end - overlap_start

    return {
        'fixation_spread': fixation_spread,
        'object_looking_duration': object_looking_duration
    }

def calculate_teleport_metrics(interactions_data: List[Dict], window_start: float, window_end: float) -> Dict[str, float]:
    """Calculate teleport propensity and average stop duration within the time window."""
    # Get teleports in this window
    teleport_times = []
    for interaction in interactions_data:
        if (interaction['interaction'] == 'Teleport event' and
            window_start <= interaction['time'] < window_end):
            teleport_times.append(interaction['time'])

    # Sort teleport times
    teleport_times.sort()

    # Calculate teleport propensity (rate per second)
    window_duration = window_end - window_start
    teleport_propensity = len(teleport_times) / window_duration if window_duration > 0 else 0.0

    # Calculate average stop duration
    if len(teleport_times) > 1:
        # Calculate intervals between consecutive teleports
        stop_durations = []
        for i in range(1, len(teleport_times)):
            stop_duration = teleport_times[i] - teleport_times[i-1]
            stop_durations.append(stop_duration)

        # Average of all stop durations
        avg_stop_duration = sum(stop_durations) / len(stop_durations) if stop_durations else 0.0
    elif len(teleport_times) == 1:
        # For single teleport, use max distance from window boundaries
        teleport_time = teleport_times[0]
        avg_stop_duration = max(abs(window_start - teleport_time), abs(window_end - teleport_time))
    else:
        # For no teleports, use full window duration (5 seconds)
        avg_stop_duration = 5.0

    return {
        'teleport_propensity': teleport_propensity,
        'avg_stop_duration': avg_stop_duration
    }

def process_subject_session(data_dir: str, subject_id: str, session: str,
                           object_positions: Dict, age_info: Dict, interaction_index_offset: int = 0) -> List[Dict]:
    """Process all MS events for a single subject session."""
    # Find files for this subject and session
    pattern = f"S-{subject_id}_{session}_*"
    matching_files = glob.glob(os.path.join(data_dir, pattern))

    head_file = next((f for f in matching_files if 'head-tracking' in f), None)
    gaze_events_file = next((f for f in matching_files if 'gaze-events' in f), None)
    gaze_vectors_file = next((f for f in matching_files if 'gaze-vectors' in f), None)
    interaction_file = next((f for f in matching_files if 'interactions' in f), None)

    if not all([head_file, gaze_events_file, gaze_vectors_file, interaction_file]):
        logging.warning(f"Missing files for {subject_id} {session}")
        return []

    # Extract standardized subject ID from filename
    standardized_subject_id = extract_subject_id_from_filename(head_file)
    age_group = determine_age_group_from_path(head_file)

    # Read data
    head_data = read_head_tracking_file(head_file)
    gaze_events_data = read_gaze_events_file(gaze_events_file)
    gaze_vectors_data = read_gaze_vectors_file(gaze_vectors_file)
    ms_events, all_interactions = read_interactions_file(interaction_file)

    if head_data.empty or len(ms_events) == 0 or gaze_vectors_data.empty:
        logging.warning(f"No data for {subject_id} {session}")
        return []

    results = []
    encounter_counts = {}  # Track encounters per object

    for local_index, event in enumerate(ms_events):
        interaction_index = interaction_index_offset + local_index
        object_name = event['object_name']
        event_time = event['time']

        # Track encounter number
        if object_name not in encounter_counts:
            encounter_counts[object_name] = 0
        encounter_counts[object_name] += 1

        # Get object position
        if object_name not in object_positions:
            logging.warning(f"Object {object_name} not found in positions file")
            continue

        object_pos = object_positions[object_name]

        # Create two windows: [-5,0) and [0,5]
        windows = [
            (event_time - 5, event_time, 0),  # Pre-interaction window, window_index=0
            (event_time, event_time + 5, 1)   # Post-interaction window, window_index=1
        ]

        for window_start, window_end, window_index in windows:
            # Calculate all metrics
            distance_metrics = calculate_distance_metrics(head_data, object_pos, window_start, window_end)
            head_metrics = calculate_head_metrics(head_data, window_start, window_end)
            gaze_metrics = calculate_gaze_metrics(gaze_vectors_data, gaze_events_data, object_name, window_start, window_end, subject_id)
            teleport_metrics = calculate_teleport_metrics(all_interactions, window_start, window_end)

            # Get age information
            age = age_info.get(standardized_subject_id, np.nan)

            result = {
                # Window identification
                'window_start': window_start,
                'window_end': window_end,
                'object_name': object_name,
                'object_pos_x': object_pos[0],
                'object_pos_z': object_pos[1],

                # Head movement metrics
                **head_metrics,

                # Gaze metrics
                **gaze_metrics,

                # Distance metrics
                **distance_metrics,

                # Navigation metrics
                **teleport_metrics,

                # Metadata
                'subject_id': standardized_subject_id,
                'session': session.lower(),
                'interaction_index': interaction_index,
                'window_index': window_index,
                'MSevent': True,
                'encounterNum': encounter_counts[object_name],

                # Demographics
                'Age': age,
                'AgeGroup': age_group
            }

            results.append(result)

    return results

def process_subject_all_sessions(data_dir: str, subject_id: str, object_positions: Dict, age_info: Dict) -> List[Dict]:
    """Process all sessions (learn1, learn2) for a subject with continuous interaction indexing."""
    all_results = []
    interaction_index_offset = 0

    # Process sessions in order: learn1, then learn2
    for session in ['learn1', 'learn2']:
        session_results = process_subject_session(data_dir, subject_id, session,
                                                object_positions, age_info, interaction_index_offset)
        all_results.extend(session_results)

        # Update offset for next session (each event creates 2 records for 2 windows)
        if session_results:
            num_events_this_session = len(session_results) // 2  # 2 records per event
            interaction_index_offset += num_events_this_session

    return all_results

def main():
    parser = argparse.ArgumentParser(description='Process MS events with simplified 2-window structure')
    parser.add_argument('--children_dir',
                       default='Children',
                       help='Path to children data directory')
    parser.add_argument('--adults_dir',
                       default='Adults',
                       help='Path to adults data directory')
    parser.add_argument('--distractors_file',
                       default='DA/scripts/distractor_files/distractors_file_medium.csv',
                       help='Path to distractors file')
    parser.add_argument('--age_file',
                       default='DA/scripts/cleaning/0115_clean/ageInfo.csv',
                       help='Path to age info file')
    parser.add_argument('--output_file',
                       default='DA/scripts/cleaning/0115_clean/ms_events_simplified.csv',
                       help='Path to output CSV file')

    args = parser.parse_args()

    # Load object positions and age info
    object_positions = load_object_positions(args.distractors_file)
    age_info = load_age_info(args.age_file)

    all_results = []

    # Process children data
    if os.path.exists(args.children_dir):
        logging.info("Processing children data...")
        # Get unique subjects from files
        children_subjects = set()
        for file in glob.glob(os.path.join(args.children_dir, "S-*interactions*")):
            if '_learn' in file.lower():  # Only process learn sessions
                subject_id = extract_original_subject_id_from_filename(file)
                children_subjects.add(subject_id)

        for subject_id in sorted(children_subjects):
            results = process_subject_all_sessions(args.children_dir, subject_id, object_positions, age_info)
            all_results.extend(results)

    # Process adults data
    if os.path.exists(args.adults_dir):
        logging.info("Processing adults data...")
        # Get unique subjects from files
        adults_subjects = set()
        for file in glob.glob(os.path.join(args.adults_dir, "S-*interactions*")):
            if '_learn' in file.lower():  # Only process learn sessions
                subject_id = extract_original_subject_id_from_filename(file)
                adults_subjects.add(subject_id)

        for subject_id in sorted(adults_subjects):
            results = process_subject_all_sessions(args.adults_dir, subject_id, object_positions, age_info)
            all_results.extend(results)

    # Save results
    if all_results:
        df = pd.DataFrame(all_results)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        df.to_csv(args.output_file, index=False)

        logging.info(f"Saved {len(df)} records to {args.output_file}")
        logging.info(f"Unique subjects: {df['subject_id'].nunique()}")
        logging.info(f"Unique objects: {df['object_name'].nunique()}")
        # Each MS event has 2 windows, so total events = records / 2
        total_ms_events = len(df) // 2
        logging.info(f"MS events: {total_ms_events}")
    else:
        logging.warning("No results to save")

if __name__ == "__main__":
    main()