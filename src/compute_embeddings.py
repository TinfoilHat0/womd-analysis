"""
Compute scene-level embeddings from WOMD motion data using MOMENT-1 time series encoder.

For each scenario:
1. Normalize all agent trajectories to SDC-centered coordinates
2. Encode each agent's trajectory [91 timesteps × 4 channels] → 768-dim embedding
3. Mean-pool across agents → single scene embedding

Usage:
    python src/compute_embeddings.py [--model MOMENT-1-base] [--device mps|cuda|cpu]
                                     [--batch-size 32] [--max-agents 64]
                                     [--data-dir data/womd_motion]
                                     [--output data/embeddings.npz]
"""

import argparse
import glob
import math
import os
import sys
import time

import numpy as np
import torch
from momentfm import MOMENTPipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.feature_extraction import read_tfrecord, parse_scenario


# ---------------------------------------------------------------------------
# Trajectory extraction
# ---------------------------------------------------------------------------

def extract_agent_trajectories(scenario, max_agents=64, min_valid=10):
    """
    Extract SDC-centered trajectories for all valid agents.

    Returns:
        trajs: np.array [N_agents, 4, 91] — (x, y, vx, vy) per timestep
        agent_types: list of int — object_type per agent
        n_valid: int — number of valid agents
    """
    s = scenario
    sdc = s.tracks[s.sdc_track_index]
    ct = s.current_time_index
    sdc_st = sdc.states[ct]
    cx, cy = sdc_st.center_x, sdc_st.center_y
    cos_h = math.cos(-sdc_st.heading)
    sin_h = math.sin(-sdc_st.heading)

    # Sort agents by distance to SDC at current_time_index (closest first)
    agent_dists = []
    for i, track in enumerate(s.tracks):
        st = track.states[ct]
        if st.valid:
            d = math.sqrt((st.center_x - cx) ** 2 + (st.center_y - cy) ** 2)
        else:
            d = float('inf')
        agent_dists.append((i, d))
    agent_dists.sort(key=lambda x: x[1])

    trajs = []
    agent_types = []

    for track_idx, _ in agent_dists:
        if len(trajs) >= max_agents:
            break

        track = s.tracks[track_idx]
        traj = np.zeros((91, 4), dtype=np.float32)
        valid_count = 0

        for t in range(91):
            st = track.states[t]
            if st.valid:
                dx = st.center_x - cx
                dy = st.center_y - cy
                traj[t, 0] = dx * cos_h - dy * sin_h
                traj[t, 1] = dx * sin_h + dy * cos_h
                traj[t, 2] = st.velocity_x * cos_h - st.velocity_y * sin_h
                traj[t, 3] = st.velocity_x * sin_h + st.velocity_y * cos_h
                valid_count += 1

        if valid_count < min_valid:
            continue

        trajs.append(traj.T)  # [4, 91]
        agent_types.append(track.object_type)

    if not trajs:
        return None, None, 0

    return np.stack(trajs), agent_types, len(trajs)


# ---------------------------------------------------------------------------
# MOMENT encoding
# ---------------------------------------------------------------------------

def encode_agents_batched(model, trajs, device, batch_size=32):
    """
    Encode agent trajectories using MOMENT.

    Args:
        trajs: np.array [N_agents, 4, 91]

    Returns:
        embeddings: np.array [N_agents, embed_dim]
    """
    n_agents = trajs.shape[0]

    # Pad from 91 to 512 timesteps (MOMENT's expected length)
    padded = np.zeros((n_agents, 4, 512), dtype=np.float32)
    padded[:, :, :91] = trajs

    input_mask = np.zeros((n_agents, 512), dtype=np.float32)
    input_mask[:, :91] = 1.0

    all_embs = []
    for i in range(0, n_agents, batch_size):
        batch_t = torch.tensor(padded[i:i + batch_size]).to(device)
        batch_m = torch.tensor(input_mask[i:i + batch_size]).to(device)

        with torch.no_grad():
            output = model(x_enc=batch_t, input_mask=batch_m)

        all_embs.append(output.embeddings.cpu().numpy())

    return np.concatenate(all_embs, axis=0)


def compute_scene_embedding(agent_embeddings):
    """Mean-pool agent embeddings to get a single scene embedding."""
    return agent_embeddings.mean(axis=0)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_all_scenarios(model, device, data_dir, batch_size=32,
                          max_agents=64):
    """Process all TFRecord files and return scene embeddings."""
    shard_files = sorted(glob.glob(os.path.join(data_dir, '*.tfrecord*')))
    if not shard_files:
        raise FileNotFoundError(f'No TFRecord files found in {data_dir}')

    scenario_ids = []
    scene_embeddings = []
    agent_counts = []

    total_scenarios = 0
    t_start = time.time()

    for shard_idx, shard_path in enumerate(shard_files):
        shard_name = os.path.basename(shard_path)
        records = read_tfrecord(shard_path)

        for raw in records:
            scenario = parse_scenario(raw)

            trajs, agent_types, n_agents = extract_agent_trajectories(
                scenario, max_agents=max_agents
            )

            if trajs is None:
                continue

            agent_embs = encode_agents_batched(
                model, trajs, device, batch_size=batch_size
            )
            scene_emb = compute_scene_embedding(agent_embs)

            scenario_ids.append(scenario.scenario_id)
            scene_embeddings.append(scene_emb)
            agent_counts.append(n_agents)
            total_scenarios += 1

        elapsed = time.time() - t_start
        rate = total_scenarios / elapsed if elapsed > 0 else 0
        print(f'  [{shard_idx + 1}/{len(shard_files)}] {shard_name}: '
              f'{total_scenarios} scenarios so far '
              f'({rate:.1f} scenarios/s, {elapsed:.0f}s elapsed)')

    return scenario_ids, np.stack(scene_embeddings), agent_counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Compute MOMENT embeddings for WOMD scenarios')
    parser.add_argument('--model', default='AutonLab/MOMENT-1-base',
                        help='HuggingFace model name (default: MOMENT-1-base)')
    parser.add_argument('--device', default=None,
                        help='Device: cuda, mps, or cpu (auto-detected if not set)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Agents per batch for encoding (default: 32)')
    parser.add_argument('--max-agents', type=int, default=64,
                        help='Max agents per scenario, closest to SDC (default: 64)')
    parser.add_argument('--data-dir', default='data/womd_motion',
                        help='Directory with TFRecord files')
    parser.add_argument('--output', default='data/embeddings.npz',
                        help='Output path for embeddings (.npz)')
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'

    device = torch.device(args.device)
    print(f'Device: {device}')
    print(f'Model: {args.model}')
    print(f'Batch size: {args.batch_size}')
    print(f'Max agents per scenario: {args.max_agents}')
    print(f'Data dir: {args.data_dir}')
    print(f'Output: {args.output}')
    print()

    # Load model
    print('Loading model...')
    model = MOMENTPipeline.from_pretrained(
        args.model,
        model_kwargs={'task_name': 'embedding'}
    )
    model.init()
    model = model.to(device)
    print('Model loaded.\n')

    # Process
    print('Processing scenarios...')
    scenario_ids, embeddings, agent_counts = process_all_scenarios(
        model, device, args.data_dir,
        batch_size=args.batch_size,
        max_agents=args.max_agents,
    )

    # Save
    np.savez(
        args.output,
        scenario_ids=np.array(scenario_ids),
        embeddings=embeddings,
        agent_counts=np.array(agent_counts),
    )
    print(f'\nSaved {len(scenario_ids)} embeddings ({embeddings.shape}) to {args.output}')


if __name__ == '__main__':
    main()
