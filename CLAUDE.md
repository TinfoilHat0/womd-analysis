# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project for driving scenario embedding and rare event detection using the Waymo Open Motion Dataset (WOMD). The goal is to cluster autonomous driving scenarios by hand-crafted features and learned embeddings to identify rare/interesting events — aligned with Zoox's validation pipeline needs.

**Three-phase plan:** (1) Hand-crafted features + clustering (done), (2) Learned trajectory embeddings (in progress), (3) Retrieval and Bayesian search for rare events.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Scripts

```bash
# Feature extraction from raw TFRecord data → data/features.json
python src/feature_extraction.py

# Clustering pipeline (K-means, HDBSCAN, GMM with PCA + UMAP visualization)
python src/clustering.py

# BEV rendering of scenarios (PNGs + animated GIFs)
python src/render_bev.py

# Compute MOMENT-1 embeddings → data/embeddings.npz
python src/compute_embeddings.py [--device cuda|mps|cpu] [--batch-size 32] [--max-agents 64]
```

## Dependencies

See `requirements.txt`. Key packages: numpy, scikit-learn, hdbscan, umap-learn, matplotlib, protobuf, torch, momentfm.

## Architecture

### Data Flow
```
Raw TFRecords (data/womd_motion/)
  → feature_extraction.py → data/features.json (26 hand-crafted features)
  → compute_embeddings.py → data/embeddings.npz (768-dim MOMENT embeddings)
  → clustering.py → cluster plots + profiles in data/
  → render_bev.py → BEV renders in data/bev_renders/
```

### Key Modules
- **`src/feature_extraction.py`** — Reads TFRecords without TensorFlow (custom binary reader), parses Scenario protobufs, extracts 26 features per scenario across 5 categories (ego dynamics, safety/risk, interaction, map context, temporal).
- **`src/clustering.py`** — Loads `features.json`, preprocesses (drop correlated, binarize, log-transform, z-score), runs K-means + HDBSCAN + GMM, projects via PCA and UMAP, outputs plots and cluster profiles.
- **`src/compute_embeddings.py`** — Encodes agent trajectories using MOMENT-1 time series model. Per agent: normalize to SDC-centered coords, encode [91 timesteps × 4 channels] → 768-dim. Mean-pool across agents → scene embedding. Auto-detects CUDA/MPS/CPU.
- **`src/render_bev.py`** — BEV renderer for scenarios. Draws map features (lanes, crosswalks, stop signs), agent trajectories (history solid, future dashed), bounding boxes. Supports static PNGs and animated GIFs with follow-camera.
- **`waymo_open_dataset/`** — Pre-compiled protobuf Python bindings. Key proto: `scenario_pb2.py`. Do not regenerate.

### Data Layout
- `data/womd_motion/` — 50 TFRecord shards (14,441 scenarios), validation_interactive split
- `data/womd_reasoning/` — Language Q&A JSONs (explored, too templated — not used)
- `data/causal_agents/` — Human-labeled causal agent annotations (43K scenarios, potential ground truth)
- `data/features.json` — Hand-crafted features (14,441 × 26)
- `data/embeddings.npz` — MOMENT scene embeddings (14,441 × 768)
- `data/bev_renders/` — Rendered scenario visualizations (PNGs + GIFs)

### Proto Access Pattern
The TFRecord reader in `feature_extraction.py` is a pure-Python implementation (no TensorFlow dependency). It reads length-prefixed records and parses them with `scenario_pb2.Scenario().ParseFromString()`.

## Key Domain Concepts
- **SDC** = Self-Driving Car (ego vehicle), identified by `scenario.sdc_track_index`
- **TTC** = Time To Collision (distance / closing speed between agent pairs)
- Scenarios are 9 seconds at 10Hz (91 timesteps), with `current_time_index` splitting history from future
- Agent types: VEHICLE=1, PEDESTRIAN=2, CYCLIST=3
- Features use `valid` bit filtering — not all agents are observed at all timesteps
- Coordinates are normalized to SDC-centered frame for embeddings (translate + rotate so SDC at origin facing +x at current_time_index)
