#!/usr/bin/env python3
"""
Corrected Spatial Backtracking Detector - Clean Implementation

Detects backtracking episodes in spatial navigation data using strict spatial validation.
Preserves exact algorithm logic from original CorrectedSpatialDetector.

Key Features:
- Distance-based adaptive subsampling (5m target intervals)
- Bidirectional spatial overlap validation (80% threshold)
- Strict turning point detection (120° minimum angle)
- Area constraints (400m² maximum episode area)
- Balanced approach/return phases (30%-70% ratios)

Author: Recreated from CorrectedSpatialDetector
Date: 2026-02-01
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpatialBacktrackingDetector:
    """
    Corrected spatial backtracking detector with strict validation criteria.

    This detector implements the exact algorithm logic from the original
    CorrectedSpatialDetector with cleaner, more maintainable code structure.
    """

    def __init__(self):
        """Initialize detector with exact parameters from original implementation."""
        # Subsampling parameters
        self.target_distance = 5.0  # Target distance between subsampled sequences

        # Episode structure requirements
        self.min_approach_sequences = 4  # Minimum sequences before turning point
        self.min_return_sequences = 4   # Minimum sequences after turning point

        # Balance ratio constraints
        self.min_balance_ratio = 3/10    # 30% minimum for each phase
        self.max_balance_ratio = 7/10    # 70% maximum for each phase

        # Spatial validation constraints
        self.max_episode_area = 400      # Maximum 400 m² (20m x 20m area)
        self.min_spatial_overlap = 0.8   # 80% of return sequences must be near approach area
        self.proximity_threshold = 8.0   # Return sequences must be within 8m of approach area

        # Turning point detection
        self.min_turning_angle = 120     # Minimum 120° turn for clear direction change

        logger.info("SpatialBacktrackingDetector initialized with corrected parameters")

    def load_navigation_data(self, file_path: str, subject_id: str) -> dict:
        """
        Load and preprocess navigation data from head-tracking files.

        Args:
            file_path: Path to head-tracking.txt file
            subject_id: Subject identifier

        Returns:
            Dictionary with positions, timestamps, and metadata
        """
        logger.info(f"Loading navigation data for {subject_id}")
        logger.info(f"File: {Path(file_path).name}")

        try:
            # Read file to find header line
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Find column headers (contains 'pos')
            header_line = None
            for i, line in enumerate(lines):
                if 'pos' in line.lower() and ('x' in line.lower() or 'z' in line.lower()):
                    header_line = i
                    break

            if header_line is None:
                logger.error("Could not find data header in file")
                return None

            # Load data starting from header line
            df = pd.read_csv(file_path, sep='\t', skiprows=header_line)
            df.columns = df.columns.str.strip()

            # Find position columns
            pos_x_col = self._find_column(df.columns, ['pos', 'x'])
            pos_z_col = self._find_column(df.columns, ['pos', 'z'])
            time_col = self._find_column(df.columns, ['second']) or self._find_column(df.columns, ['time'])

            if pos_x_col is None or pos_z_col is None:
                logger.error(f"Could not find position columns. Available: {list(df.columns)}")
                return None

            logger.info(f"Found position columns: {pos_x_col}, {pos_z_col}")

            # Clean and validate data
            df = df.dropna(subset=[pos_x_col, pos_z_col])
            df[pos_x_col] = pd.to_numeric(df[pos_x_col], errors='coerce')
            df[pos_z_col] = pd.to_numeric(df[pos_z_col], errors='coerce')
            df = df.dropna(subset=[pos_x_col, pos_z_col])

            # Remove origin points (optional)
            df = df[(df[pos_x_col] != 0) | (df[pos_z_col] != 0)]

            if len(df) < 50:
                logger.error(f"Insufficient data points: {len(df)} < 50")
                return None

            positions = df[[pos_x_col, pos_z_col]].values
            timestamps = df[time_col].values if time_col and time_col in df.columns else np.arange(len(df))

            logger.info(f"Loaded {len(positions)} valid position points")

            return {
                'positions': positions,
                'timestamps': timestamps,
                'subject_id': subject_id,
                'file_path': file_path
            }

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

    def _find_column(self, columns, keywords):
        """Find column containing all keywords (case insensitive)."""
        for col in columns:
            col_lower = col.lower()
            if all(keyword.lower() in col_lower for keyword in keywords):
                return col
        return None

    def distance_based_subsampling(self, positions: np.ndarray) -> tuple:
        """
        Apply distance-based adaptive subsampling.

        Selects points based on 5m movement distance rather than time intervals.
        This prevents oversampling in stationary areas while maintaining spatial resolution.

        Args:
            positions: Array of (x, z) position coordinates

        Returns:
            Tuple of (selected_indices, selected_positions, movement_distances)
        """
        logger.info("Applying distance-based subsampling")
        logger.info(f"Target distance: {self.target_distance}m between sequences")

        if len(positions) < 3:
            return [], [], []

        # Initialize with first position
        selected_indices = [0]
        selected_positions = [positions[0]]
        movement_distances = []
        current_pos = positions[0]

        # Iterate through all positions
        for i in range(1, len(positions)):
            next_pos = positions[i]
            distance = np.sqrt(np.sum((next_pos - current_pos) ** 2))

            # Selection criteria: distance threshold OR safety fallback
            should_select = (distance >= self.target_distance or
                           (i - selected_indices[-1]) >= 100)

            # Ensure minimum spacing between selections
            if should_select and (i - selected_indices[-1]) >= 3:
                selected_indices.append(i)
                selected_positions.append(next_pos)
                movement_distances.append(distance)
                current_pos = next_pos

        selected_positions = np.array(selected_positions)
        movement_distances = np.array(movement_distances)

        # Log subsampling results
        compression_ratio = len(selected_positions) / len(positions) * 100
        avg_distance = np.mean(movement_distances) if len(movement_distances) > 0 else 0

        logger.info(f"Subsampling results:")
        logger.info(f"  Original points: {len(positions)}")
        logger.info(f"  Selected sequences: {len(selected_positions)}")
        logger.info(f"  Compression ratio: {compression_ratio:.1f}%")
        logger.info(f"  Average distance between sequences: {avg_distance:.1f}m")

        return selected_indices, selected_positions, movement_distances

    def calculate_bidirectional_spatial_overlap(self, approach_positions: np.ndarray,
                                               return_positions: np.ndarray) -> tuple:
        """
        Calculate bidirectional spatial overlap between approach and return paths.

        This is the core spatial validation - both phases must overlap spatially
        to constitute true backtracking behavior.

        Args:
            approach_positions: Positions before turning point
            return_positions: Positions after turning point

        Returns:
            Tuple of (bidirectional_overlap, return_overlap_ratio, approach_overlap_ratio)
        """
        if len(approach_positions) == 0 or len(return_positions) == 0:
            return 0.0, 0.0, 0.0

        # Calculate distance matrix between all approach and return positions
        distances = cdist(return_positions, approach_positions)

        # Direction 1: Return sequences overlapping with approach area
        min_distances_return = np.min(distances, axis=1)
        close_return_positions = np.sum(min_distances_return <= self.proximity_threshold)
        return_overlap_ratio = close_return_positions / len(return_positions)

        # Direction 2: Approach sequences overlapping with return area
        min_distances_approach = np.min(distances, axis=0)
        close_approach_positions = np.sum(min_distances_approach <= self.proximity_threshold)
        approach_overlap_ratio = close_approach_positions / len(approach_positions)

        # Bidirectional requirement: both phases must meet overlap threshold
        bidirectional_overlap = min(return_overlap_ratio, approach_overlap_ratio)

        return bidirectional_overlap, return_overlap_ratio, approach_overlap_ratio

    def detect_turning_point(self, positions: np.ndarray, start_idx: int,
                           end_idx: int, detailed_output: bool = False) -> dict:
        """
        Detect optimal turning point within a segment with spatial validation.

        The turning point must satisfy:
        - Minimum turning angle (120°)
        - Balanced approach/return phases
        - Bidirectional spatial overlap

        Args:
            positions: Array of sequence positions
            start_idx: Start index of segment
            end_idx: End index of segment
            detailed_output: Whether to log detailed validation steps

        Returns:
            Dictionary with turning point information or None if invalid
        """
        if end_idx - start_idx < 8:  # Need minimum 8 points for 4+4
            return None

        segment = positions[start_idx:end_idx+1]
        best_turn = None
        best_score = 0

        if detailed_output:
            logger.info(f"Testing turning points for segment {start_idx+1}-{end_idx+1}")

        # Test potential turning points (ensure 4+ before and after)
        for turn_idx in range(start_idx + 4, end_idx - 3):
            local_idx = turn_idx - start_idx

            if local_idx < 4 or local_idx >= len(segment) - 4:
                continue

            # Calculate directional change at turning point
            if local_idx >= 2 and local_idx < len(segment) - 2:
                # Direction vectors before and after turning point
                vec_before = segment[local_idx] - segment[local_idx-2]
                vec_after = segment[local_idx+2] - segment[local_idx]

                # Calculate angle between vectors
                cos_angle = np.dot(vec_before, vec_after) / (
                    np.linalg.norm(vec_before) * np.linalg.norm(vec_after) + 1e-8)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle_deg = np.degrees(np.arccos(cos_angle))

                # Require significant turning angle
                if angle_deg < self.min_turning_angle:
                    if detailed_output:
                        logger.debug(f"Turn at {turn_idx+1}: {angle_deg:.1f}° < {self.min_turning_angle}°")
                    continue

                # Check balance: sequences before vs after turning point
                approach_sequences = turn_idx - start_idx
                return_sequences = end_idx - turn_idx
                total_sequences = end_idx - start_idx + 1

                approach_ratio = approach_sequences / total_sequences
                return_ratio = return_sequences / total_sequences

                # Validate balance ratios
                balance_valid = (approach_sequences >= self.min_approach_sequences and
                               return_sequences >= self.min_return_sequences and
                               approach_ratio >= self.min_balance_ratio and
                               return_ratio >= self.min_balance_ratio and
                               approach_ratio <= self.max_balance_ratio and
                               return_ratio <= self.max_balance_ratio)

                if not balance_valid:
                    if detailed_output:
                        logger.debug(f"Turn at {turn_idx+1}: Imbalanced phases")
                    continue

                # Bidirectional spatial overlap validation
                approach_positions = segment[:local_idx+1]  # Include turning point
                return_positions = segment[local_idx:]      # Include turning point

                bidirectional_overlap, return_overlap, approach_overlap = \
                    self.calculate_bidirectional_spatial_overlap(approach_positions, return_positions)

                if bidirectional_overlap < self.min_spatial_overlap:
                    if detailed_output:
                        logger.debug(f"Turn at {turn_idx+1}: Insufficient spatial overlap")
                    continue

                # Calculate combined score
                angle_score = min(angle_deg / 180.0, 1.0)
                spatial_score = bidirectional_overlap
                balance_score = 1.0 - abs(approach_ratio - 0.5) * 2  # Prefer balanced splits

                combined_score = angle_score * spatial_score * balance_score

                if combined_score > best_score:
                    best_score = combined_score
                    best_turn = {
                        'turning_point_seq': turn_idx,
                        'angle_deg': angle_deg,
                        'approach_sequences': approach_sequences,
                        'return_sequences': return_sequences,
                        'approach_ratio': approach_ratio,
                        'return_ratio': return_ratio,
                        'spatial_overlap': bidirectional_overlap,
                        'return_overlap': return_overlap,
                        'approach_overlap': approach_overlap,
                        'combined_score': combined_score
                    }

                if detailed_output:
                    logger.debug(f"Turn at {turn_idx+1}: Valid (Score: {combined_score:.2f})")

        return best_turn

    def validate_spatial_episode(self, positions: np.ndarray, start_idx: int,
                               end_idx: int, turning_point_info: dict) -> tuple:
        """
        Validate episode spatial constraints.

        Enforces area constraints and confirms spatial overlap requirements.

        Args:
            positions: Array of sequence positions
            start_idx: Episode start index
            end_idx: Episode end index
            turning_point_info: Turning point validation results

        Returns:
            Tuple of (is_valid, validation_message)
        """
        if turning_point_info is None:
            return False, "No valid turning point found"

        segment_positions = positions[start_idx:end_idx+1]

        # Calculate episode area
        if len(segment_positions) >= 3:
            try:
                hull = ConvexHull(segment_positions)
                area = hull.volume  # In 2D, volume gives area
            except Exception:
                # Fallback to bounding box if convex hull fails
                x_range = np.max(segment_positions[:, 0]) - np.min(segment_positions[:, 0])
                z_range = np.max(segment_positions[:, 1]) - np.min(segment_positions[:, 1])
                area = x_range * z_range

            if area > self.max_episode_area:
                return False, f"Episode area too large: {area:.1f}m² > {self.max_episode_area}m²"

        # Spatial overlap already validated in turning point detection
        overlap_ratio = turning_point_info['spatial_overlap']
        approach_sequences = turning_point_info['approach_sequences']
        return_sequences = turning_point_info['return_sequences']

        return True, f"Valid episode: {approach_sequences}+{return_sequences} sequences, " \
                     f"{overlap_ratio:.1%} spatial overlap, {area:.1f}m² area"

    def detect_episodes(self, positions: np.ndarray) -> list:
        """
        Detect backtracking episodes with spatial validation.

        Searches for valid episodes by testing different segment lengths
        and validating spatial overlap requirements.

        Args:
            positions: Array of subsampled sequence positions

        Returns:
            List of valid episode dictionaries
        """
        logger.info("Detecting spatial backtracking episodes")
        logger.info(f"Analyzing {len(positions)} spatially distributed positions")

        episodes = []
        min_episode_length = self.min_approach_sequences + self.min_return_sequences + 1  # 9 sequences

        # Search for episodes of different lengths
        for start_idx in range(len(positions) - min_episode_length + 1):
            for length in range(min_episode_length, min(len(positions) - start_idx + 1, 31)):
                end_idx = start_idx + length - 1

                logger.debug(f"Testing segment {start_idx+1}-{end_idx+1} ({length} sequences)")

                # Detect turning point with spatial validation
                turning_point_info = self.detect_turning_point(
                    positions, start_idx, end_idx, detailed_output=False)

                if turning_point_info is None:
                    continue

                # Validate spatial constraints
                is_valid, validation_msg = self.validate_spatial_episode(
                    positions, start_idx, end_idx, turning_point_info)

                if not is_valid:
                    logger.debug(f"Spatial validation failed: {validation_msg}")
                    continue

                logger.info(f"Valid backtracking episode: {validation_msg}")

                # Create episode record
                episode = {
                    'start_seq': start_idx + 1,  # 1-based indexing
                    'end_seq': end_idx + 1,
                    'turning_point_seq': turning_point_info['turning_point_seq'] + 1,
                    'length': length,
                    'turning_angle_deg': turning_point_info['angle_deg'],
                    'approach_sequences': turning_point_info['approach_sequences'],
                    'return_sequences': turning_point_info['return_sequences'],
                    'approach_ratio': turning_point_info['approach_ratio'],
                    'return_ratio': turning_point_info['return_ratio'],
                    'spatial_overlap': turning_point_info['spatial_overlap'],
                    'return_overlap': turning_point_info['return_overlap'],
                    'approach_overlap': turning_point_info['approach_overlap'],
                    'combined_score': turning_point_info['combined_score'],
                    'validation_msg': validation_msg
                }

                episodes.append(episode)

        logger.info(f"Valid backtracking episodes found: {len(episodes)}")
        return episodes

    def filter_overlapping_episodes(self, episodes: list) -> list:
        """
        Remove overlapping episodes, keeping highest scoring ones.

        Args:
            episodes: List of episode dictionaries

        Returns:
            List of non-overlapping episodes sorted by start sequence
        """
        if len(episodes) <= 1:
            return episodes

        logger.info(f"Filtering overlapping episodes (input: {len(episodes)})")

        # Sort by combined score (descending)
        episodes_sorted = sorted(episodes, key=lambda x: x['combined_score'], reverse=True)
        filtered_episodes = []

        for episode in episodes_sorted:
            # Check for overlap with accepted episodes
            overlaps = False
            for accepted in filtered_episodes:
                if not (episode['end_seq'] < accepted['start_seq'] or
                       episode['start_seq'] > accepted['end_seq']):
                    overlaps = True
                    logger.debug(f"Rejecting overlapping episode {episode['start_seq']}-{episode['end_seq']}")
                    break

            if not overlaps:
                filtered_episodes.append(episode)
                logger.debug(f"Accepting episode {episode['start_seq']}-{episode['end_seq']}")

        logger.info(f"Final episodes after overlap filtering: {len(filtered_episodes)}")
        return sorted(filtered_episodes, key=lambda x: x['start_seq'])

    def analyze_subject(self, file_path: str, subject_id: str) -> dict:
        """
        Complete analysis for a single subject.

        Performs the full pipeline: data loading, subsampling, episode detection,
        and overlap filtering.

        Args:
            file_path: Path to head-tracking data file
            subject_id: Subject identifier

        Returns:
            Dictionary with analysis results or None if failed
        """
        logger.info(f"Starting spatial backtracking analysis for {subject_id}")
        logger.info("=" * 60)

        # Load data
        data = self.load_navigation_data(file_path, subject_id)
        if data is None:
            logger.error(f"Failed to load data for {subject_id}")
            return None

        positions = data['positions']

        # Distance-based subsampling
        subsample_indices, subsample_positions, movement_distances = \
            self.distance_based_subsampling(positions)

        if len(subsample_positions) < 12:  # Need minimum for 4+4+4
            logger.error(f"Insufficient subsampled data: {len(subsample_positions)} < 12 sequences")
            return None

        # Detect episodes
        candidate_episodes = self.detect_episodes(subsample_positions)

        if len(candidate_episodes) == 0:
            logger.info(f"No valid backtracking episodes detected for {subject_id}")
            return {
                'subject_id': subject_id,
                'file_path': file_path,
                'episodes': [],
                'candidate_episodes': [],
                'subsample_indices': subsample_indices,
                'subsample_positions': subsample_positions,
                'movement_distances': movement_distances,
                'target_distance': self.target_distance,
                'detection_method': 'corrected_spatial_backtracking'
            }

        # Filter overlapping episodes
        final_episodes = self.filter_overlapping_episodes(candidate_episodes)

        logger.info(f"Analysis complete for {subject_id}")
        logger.info(f"Valid backtracking episodes: {len(final_episodes)}")

        for i, episode in enumerate(final_episodes):
            logger.info(f"Episode {i+1}: Sequences {episode['start_seq']}-{episode['end_seq']} "
                       f"({episode['approach_sequences']}+{episode['return_sequences']}) "
                       f"Overlap: {episode['spatial_overlap']:.1%}")

        return {
            'subject_id': subject_id,
            'file_path': file_path,
            'episodes': final_episodes,
            'candidate_episodes': candidate_episodes,
            'subsample_indices': subsample_indices,
            'subsample_positions': subsample_positions,
            'movement_distances': movement_distances,
            'target_distance': self.target_distance,
            'detection_method': 'corrected_spatial_backtracking'
        }