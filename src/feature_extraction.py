"""
Feature extraction from WOMD raw motion data.

Extracts 26 hand-crafted features per scenario across 5 categories:
1. Ego Dynamics (7): speed stats, accel/decel, heading change, displacement
2. Safety / Risk (4): TTC, min distance, close encounters, hard braking
3. Interaction / Multi-Agent (7): agent counts/types, distances, oncoming/crossing
4. Scene / Map Context (5): lanes, crosswalks, stop signs, speed bumps, signals
5. Temporal Dynamics (3): speed change, stopped periods, agent density
"""

import struct
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from waymo_open_dataset.protos import scenario_pb2


# ---------------------------------------------------------------------------
# TFRecord reader
# ---------------------------------------------------------------------------

def read_tfrecord(path, max_records=None):
    """Read serialized proto records from a TFRecord file."""
    records = []
    with open(path, 'rb') as f:
        while True:
            len_bytes = f.read(8)
            if not len_bytes or len(len_bytes) < 8:
                break
            length = struct.unpack('Q', len_bytes)[0]
            f.read(4)  # length CRC
            data = f.read(length)
            f.read(4)  # data CRC
            records.append(data)
            if max_records and len(records) >= max_records:
                break
    return records


def parse_scenario(raw_bytes):
    """Parse raw bytes into a Scenario proto."""
    s = scenario_pb2.Scenario()
    s.ParseFromString(raw_bytes)
    return s


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(scenario):
    """
    Extract 26 features from a single Scenario proto.

    Returns a dict with scenario_id + 26 feature values.
    """
    s = scenario
    sdc = s.tracks[s.sdc_track_index]
    n_steps = len(s.timestamps_seconds)
    dt = 0.1  # 10Hz

    features = {'scenario_id': s.scenario_id}

    # ------------------------------------------------------------------
    # 1. Ego Dynamics (7 features)
    # ------------------------------------------------------------------
    speeds = []
    headings = []
    positions = []
    for st in sdc.states:
        if st.valid:
            speeds.append(math.sqrt(st.velocity_x**2 + st.velocity_y**2))
            headings.append(st.heading)
            positions.append((st.center_x, st.center_y))
        else:
            speeds.append(None)
            headings.append(None)
            positions.append(None)

    valid_speeds = [s for s in speeds if s is not None]
    features['ego_mean_speed'] = sum(valid_speeds) / len(valid_speeds)
    features['ego_max_speed'] = max(valid_speeds)
    features['ego_speed_std'] = _std(valid_speeds)

    # Acceleration / deceleration from consecutive speed changes
    max_accel = 0.0
    max_decel = 0.0
    for t in range(1, n_steps):
        if speeds[t] is not None and speeds[t - 1] is not None:
            delta = (speeds[t] - speeds[t - 1]) / dt
            if delta > max_accel:
                max_accel = delta
            if -delta > max_decel:
                max_decel = -delta  # store as positive
    features['ego_max_accel'] = max_accel
    features['ego_max_decel'] = max_decel

    # Total heading change
    total_heading_change = 0.0
    valid_headings = [(i, h) for i, h in enumerate(headings) if h is not None]
    for j in range(1, len(valid_headings)):
        dh = abs(_angle_diff(valid_headings[j][1], valid_headings[j - 1][1]))
        total_heading_change += dh
    features['ego_total_heading_change'] = total_heading_change

    # Displacement (start to end)
    valid_pos = [p for p in positions if p is not None]
    if len(valid_pos) >= 2:
        dx = valid_pos[-1][0] - valid_pos[0][0]
        dy = valid_pos[-1][1] - valid_pos[0][1]
        features['ego_displacement'] = math.sqrt(dx**2 + dy**2)
    else:
        features['ego_displacement'] = 0.0

    # ------------------------------------------------------------------
    # 2. Safety / Risk (4 features)
    # ------------------------------------------------------------------
    min_ttc = float('inf')
    min_distance = float('inf')
    close_encounter_agents = set()
    hard_brake_count = 0

    # Hard braking (from ego speeds)
    for t in range(1, n_steps):
        if speeds[t] is not None and speeds[t - 1] is not None:
            decel = (speeds[t - 1] - speeds[t]) / dt
            if decel > 4.0:
                hard_brake_count += 1

    # Min distance, TTC, close encounters vs all other agents
    for track in s.tracks:
        if track.id == sdc.id:
            continue
        for t in range(n_steps):
            sdc_st = sdc.states[t]
            oth_st = track.states[t]
            if not sdc_st.valid or not oth_st.valid:
                continue

            dx = oth_st.center_x - sdc_st.center_x
            dy = oth_st.center_y - sdc_st.center_y
            dist = math.sqrt(dx**2 + dy**2)

            # Min distance
            if dist < min_distance:
                min_distance = dist

            # Close encounters
            if dist < 5.0:
                close_encounter_agents.add(track.id)

            # TTC (only when dist > 3m to avoid already-passing situations)
            if dist > 3.0:
                rel_vx = sdc_st.velocity_x - oth_st.velocity_x
                rel_vy = sdc_st.velocity_y - oth_st.velocity_y
                closing = (rel_vx * dx + rel_vy * dy) / dist
                if closing > 1.0:
                    ttc = dist / closing
                    if 0 < ttc < min_ttc:
                        min_ttc = ttc

    features['min_ttc'] = min_ttc if min_ttc < 1e6 else None
    features['min_distance'] = min_distance if min_distance < 1e6 else None
    features['close_encounter_count'] = len(close_encounter_agents)
    features['hard_brake_count'] = hard_brake_count

    # ------------------------------------------------------------------
    # 3. Interaction / Multi-Agent (7 features)
    # ------------------------------------------------------------------
    type_counts = {'VEHICLE': 0, 'PEDESTRIAN': 0, 'CYCLIST': 0}
    for track in s.tracks:
        if track.object_type == 1:
            type_counts['VEHICLE'] += 1
        elif track.object_type == 2:
            type_counts['PEDESTRIAN'] += 1
        elif track.object_type == 3:
            type_counts['CYCLIST'] += 1

    features['num_agents'] = len(s.tracks)
    features['num_vehicles'] = type_counts['VEHICLE']
    features['num_pedestrians'] = type_counts['PEDESTRIAN']
    features['num_cyclists'] = type_counts['CYCLIST']

    # Mean nearest-agent distance (averaged over timesteps)
    nearest_dists = []
    for t in range(n_steps):
        sdc_st = sdc.states[t]
        if not sdc_st.valid:
            continue
        nearest = float('inf')
        for track in s.tracks:
            if track.id == sdc.id:
                continue
            oth_st = track.states[t]
            if not oth_st.valid:
                continue
            d = math.sqrt((sdc_st.center_x - oth_st.center_x)**2 +
                          (sdc_st.center_y - oth_st.center_y)**2)
            if d < nearest:
                nearest = d
        if nearest < 1e6:
            nearest_dists.append(nearest)
    features['mean_nearest_agent_dist'] = (
        sum(nearest_dists) / len(nearest_dists) if nearest_dists else None
    )

    # Oncoming agents: relative heading > 90 degrees at current_time_index
    ct = s.current_time_index
    sdc_heading = headings[ct] if headings[ct] is not None else 0
    num_oncoming = 0
    num_crossing = 0
    for track in s.tracks:
        if track.id == sdc.id:
            continue
        oth_st = track.states[ct]
        if not oth_st.valid:
            continue
        rel_heading = abs(_angle_diff(oth_st.heading, sdc_heading))
        if rel_heading > math.pi / 2:
            num_oncoming += 1
        # Crossing: heading difference between 45-135 degrees (roughly perpendicular)
        if math.pi / 4 < rel_heading < 3 * math.pi / 4:
            num_crossing += 1

    features['num_oncoming'] = num_oncoming
    features['num_crossing'] = num_crossing

    # ------------------------------------------------------------------
    # 4. Scene / Map Context (5 features)
    # ------------------------------------------------------------------
    num_lanes = 0
    num_crosswalks = 0
    num_stop_signs = 0
    num_speed_bumps = 0
    for mf in s.map_features:
        if mf.HasField('lane'):
            num_lanes += 1
        elif mf.HasField('crosswalk'):
            num_crosswalks += 1
        elif mf.HasField('stop_sign'):
            num_stop_signs += 1
        elif mf.HasField('speed_bump'):
            num_speed_bumps += 1

    features['num_lanes'] = num_lanes
    features['num_crosswalks'] = num_crosswalks
    features['num_stop_signs'] = num_stop_signs
    features['num_speed_bumps'] = num_speed_bumps
    features['has_traffic_signal'] = (
        len(s.dynamic_map_states) > 0
        and len(s.dynamic_map_states[0].lane_states) > 0
    )

    # ------------------------------------------------------------------
    # 5. Temporal Dynamics (3 features)
    # ------------------------------------------------------------------
    # Speed change (end - start)
    if valid_speeds:
        features['speed_change'] = valid_speeds[-1] - valid_speeds[0]
    else:
        features['speed_change'] = 0.0

    # Number of stopped periods (speed drops below 0.5 then resumes above 1.0)
    num_stopped_periods = 0
    in_stop = False
    for sp in valid_speeds:
        if sp < 0.5 and not in_stop:
            in_stop = True
            num_stopped_periods += 1
        elif sp > 1.0:
            in_stop = False
    features['num_stopped_periods'] = num_stopped_periods

    # Agent density: total valid agent-timesteps / num_timesteps
    total_valid = sum(
        1 for track in s.tracks for st in track.states if st.valid
    )
    features['agent_density'] = total_valid / n_steps

    return features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _angle_diff(a, b):
    """Signed angle difference, wrapped to [-pi, pi]."""
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def _std(values):
    """Standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def extract_from_tfrecord(path, max_records=None):
    """Extract features from all scenarios in a TFRecord file."""
    records = read_tfrecord(path, max_records=max_records)
    results = []
    for raw in records:
        scenario = parse_scenario(raw)
        features = extract_features(scenario)
        results.append(features)
    return results


def extract_from_directory(directory, max_records_per_file=None):
    """Extract features from all TFRecord files in a directory."""
    import glob
    files = sorted(glob.glob(os.path.join(directory, '*.tfrecord*')))
    all_results = []
    for f in files:
        print(f'Processing {os.path.basename(f)}...')
        results = extract_from_tfrecord(f, max_records=max_records_per_file)
        all_results.extend(results)
        print(f'  -> {len(results)} scenarios, total so far: {len(all_results)}')
    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import json

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'womd_motion')
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'features.json')

    print(f'Extracting features from {data_dir}')
    results = extract_from_directory(data_dir)

    # Convert bools to int for easier downstream use
    for r in results:
        r['has_traffic_signal'] = int(r['has_traffic_signal'])

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nSaved {len(results)} scenarios to {output_path}')
