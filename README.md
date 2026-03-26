# Driving Scenario Embedding & Rare Event Detection

Unsupervised analysis of autonomous driving scenarios from the [Waymo Open Motion Dataset](https://waymo.com/open/data/motion/) (WOMD). Extracts hand-crafted features from raw trajectory data, clusters scenarios to identify rare/dangerous events, and builds a Bayesian search system that finds rare scenarios **36x more efficiently** than random sampling.

This project was made using the Waymo Open Dataset, provided by Waymo LLC under the [Waymo Dataset License Agreement for Non-Commercial Use](https://waymo.com/open/terms).

@misc{waymo_open_dataset,
  title = {Waymo Open Dataset: An autonomous driving dataset},
  website = {\url{https://www.waymo.com/open}},
  year = {2019-2025}
}

## Key Results

- **Clustering:** GMM (k=14) identifies 4 rare scenario types including hard-braking emergencies and dense conflict zones, validated across K-Means and HDBSCAN
- **False positive discovery:** BEV visualization + causal agent labels reveal that some statistically "rare" scenarios are artifacts of noisy scalar features, not real danger
- **Retrieval:** Mahalanobis-distance nearest neighbors achieve **98.6% precision** — neighbors of dangerous scenarios are also genuinely interactive
- **Bayesian search:** GP surrogate + UCB acquisition finds **72% of the top-50 rarest scenarios** after evaluating only **2.4% of the dataset** (vs 2% for random sampling)
- **Ground-truth validation:** Human-labeled causal agent annotations confirm that our highest-interestingness cluster has **2.8x the average** causal agent count

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data

Raw data is not included (Waymo license). To download:

1. Register at [waymo.com/open](https://waymo.com/open) and accept the license
2. Download motion data (50 shards, ~13GB):
```bash
mkdir -p data/womd_motion
for i in $(seq -w 0 49); do
  gsutil cp "gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario/validation_interactive/validation_interactive.tfrecord-000${i}-of-00150" data/womd_motion/
done
```
3. Download causal agent labels:
```bash
mkdir -p data/causal_agents
gsutil cp "gs://waymo_open_dataset_causal_agents/causal_labels.tfrecord" data/causal_agents/
```

## Pipeline

```bash
# 1. Extract 26 hand-crafted features from raw TFRecords → data/features.json
python src/feature_extraction.py

# 2. Cluster scenarios (K-Means, HDBSCAN, GMM) + visualize (PCA, UMAP)
python src/clustering.py

# 3. Render BEV visualizations (PNGs + animated GIFs)
python src/render_bev.py

# 4. Retrieval, Bayesian search, and causal agent evaluation
python src/retrieval.py
```

## Project Structure

```
src/
├── feature_extraction.py   # TFRecord reader + 26 features across 5 categories
├── clustering.py           # K-Means, HDBSCAN, GMM with preprocessing + UMAP
├── render_bev.py           # BEV renderer with animated GIFs + follow camera
├── retrieval.py            # Retrieval, Bayesian search, causal label evaluation
└── compute_embeddings.py   # MOMENT-1 trajectory embeddings (explored, not used)

waymo_open_dataset/         # Pre-compiled protobuf bindings (do not regenerate)

data/                       # Not committed (download separately)
├── womd_motion/            # 50 TFRecord shards (14,441 scenarios)
├── causal_agents/          # Human-labeled causal agent annotations
├── features.json           # Extracted features (14,441 × 26)
└── *.png                   # Generated plots
```

## Features (26 per scenario)

| Category | Features |
|----------|----------|
| Ego Dynamics (7) | Mean/max speed, speed variance, max accel/decel, heading change, displacement |
| Safety / Risk (4) | Min TTC, min distance, close encounter count, hard braking |
| Interaction (7) | Agent count by type, nearest agent distance, oncoming/crossing counts |
| Map Context (5) | Lane/crosswalk/stop sign count, speed bumps, traffic signals |
| Temporal (3) | Speed change, stopped periods, agent density |

## Approach

**Phase 1 — Feature Engineering & Clustering:** Extract physics-based features from raw trajectories (no TensorFlow dependency — pure Python TFRecord reader). Preprocess (drop correlated, binarize, log-transform, z-score), then cluster with K-Means, HDBSCAN, and GMM. GMM selected k=14 via BIC and found the most structure.

**Phase 2 — Learned Embeddings (explored):** Tested MOMENT-1 time series foundation model for trajectory encoding. Did not improve over hand-crafted features for this task. Researched Forecast-MAE (ICCV 2023) as a stronger alternative for future work.

**Phase 3 — Retrieval & Search:** Built Mahalanobis-distance retrieval over the feature space. Implemented Bayesian rare event search using Gaussian Process surrogate with Upper Confidence Bound acquisition. Validated against 4,760 human-labeled causal agent annotations.

## References

- [Waymo Open Motion Dataset](https://waymo.com/open/data/motion/)
- [CausalAgents Benchmark](https://arxiv.org/abs/2207.03586) — Roelofs et al., 2022
- [Forecast-MAE](https://arxiv.org/abs/2308.09882) — Cheng et al., ICCV 2023
- [Large Scale AV Scenario Clustering](https://arxiv.org/abs/2103.16101) — Zhang et al., 2021

## License

Code is MIT licensed. The Waymo Open Dataset is subject to the [Waymo Dataset License Agreement for Non-Commercial Use](https://waymo.com/open/terms).
