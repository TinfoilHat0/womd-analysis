# Rare Event Detection in Autonomous Driving Scenarios

## 1. Introduction

Autonomous vehicle validation requires testing against a diverse set of driving scenarios, with particular emphasis on rare and safety-critical events. However, rare events are by definition sparse — brute-force search through massive datasets is impractical. This project develops a pipeline for extracting meaningful features from raw driving trajectory data, clustering scenarios to discover structure, and building an efficient Bayesian search system for rare event discovery.

We use the Waymo Open Motion Dataset (WOMD): 14,441 driving scenarios from the `validation_interactive` split, each containing 9.1 seconds of multi-agent trajectory data at 10Hz.

## 2. Data

Each scenario contains:
- **91 timesteps** at 10Hz with a `current_time_index` splitting history from future
- **Variable agents** (2–378 per scenario): vehicles, pedestrians, cyclists
- **Per agent per timestep:** position (x, y), velocity (vx, vy), heading, bounding box dimensions, validity flag
- **Map features:** lane polylines, road edges, crosswalks, stop signs, speed bumps, traffic signals

We processed 50 of 150 available TFRecord shards (~1/3 of the full dataset), reading them with a pure-Python TFRecord reader (no TensorFlow dependency).

For ground-truth evaluation, we use the CausalAgents dataset (Roelofs et al., 2022): 43,899 scenarios labeled by 5 human annotators marking which agents causally influence the self-driving car (SDC). 4,760 scenarios overlap with our dataset.

## 3. Feature Engineering

We extract 26 hand-crafted features per scenario across 5 categories:

**Ego Dynamics (7):** Mean speed, speed standard deviation, max acceleration, max deceleration, total heading change, lateral displacement, max speed. These capture the SDC's driving behavior — fast vs slow, aggressive vs smooth.

**Safety / Risk (4):** Minimum Time-to-Collision (TTC), minimum distance to any agent, close encounter count (agents within 5m), hard braking events. These are industry-standard safety metrics computed from pairwise agent-SDC kinematics.

**Interaction (7):** Total agent count, counts by type (vehicles, pedestrians, cyclists), mean nearest agent distance, oncoming agent count, crossing agent count. These characterize the scene complexity and interaction density.

**Map Context (5):** Lane count, crosswalk count, stop sign count, speed bump presence, traffic signal presence. These describe the road infrastructure.

**Temporal (3):** Speed change over scenario, number of stopped periods, agent density. These capture dynamic evolution.

### Preprocessing

Before clustering, we: (1) drop 4 features with |r| > 0.8 correlation to a kept feature, (2) binarize 3 zero-inflated count features, (3) log-transform 2 right-skewed features, and (4) z-score standardize. This yields 22 preprocessed features.

## 4. Clustering

We compare three unsupervised methods on the 22-feature space:

### K-Means (k=6)
Selected via elbow plot. Finds 6 clusters including one clear rare-event cluster (822 scenarios, 5.7%) characterized by hard braking at 4σ above mean, extreme deceleration (3σ), and high acceleration (2.3σ).

### HDBSCAN
Finds only 2 clusters + 273 noise points. The data is too continuously distributed for density-based clustering — no clear density gaps exist in the feature space. One cluster roughly matches K-Means' rare-event cluster.

### GMM (k=14, selected by BIC)
Gaussian Mixture Models find significantly more structure. BIC selects 14 components, providing soft probability assignments and per-scenario log-likelihood scores.

Four components correspond to rare/unusual scenarios:

| Component | Size | Signature |
|-----------|------|-----------|
| 8 | 570 | Hard braking (4σ) + extreme deceleration (3σ) — emergency reactions |
| 11 | 293 | High accel/decel + large min distance — flagged as evasive |
| 7 | 176 | Dense close encounters (2.6σ) + many oncoming agents — crowded conflicts |
| 10 | 109 | Extreme scene complexity (3.7σ oncoming, 3.6σ crossing) — massive intersections |

GMM adds value over K-Means by: (a) splitting the danger cluster into subtypes, (b) discovering rare scene-complexity types, and (c) providing a continuous log-likelihood rarity score.

## 5. BEV Visualization & False Positive Discovery

We built a bird's-eye-view renderer that draws map features (lanes, crosswalks, stop signs), agent bounding boxes color-coded by type (SDC in red, vehicles in blue, pedestrians in green, cyclists in yellow), and trajectory trails (solid for history, dashed for future). Animated GIFs with a follow-camera track the SDC through the full 9.1-second scenario.

We rendered the most extreme scenarios from each GMM component. **A critical finding emerged:** some scenarios flagged as "high-speed evasive" (component 11) show the SDC simply driving straight in its lane with no visible evasive maneuver.

**Root cause:** Scalar summary statistics like `ego_max_decel` collapse 91 timesteps into a single number, losing temporal context. A one-frame velocity spike (sensor noise or measurement artifact) produces the same `max_decel` value as a sustained emergency brake. The features cannot distinguish these fundamentally different situations.

This false positive problem motivates two things: (1) visual inspection of clustering results is essential, and (2) learned embeddings that see full trajectory shapes could potentially reduce false positives.

## 6. Learned Embeddings (Explored)

We tested MOMENT-1, a pretrained time series foundation model, as an alternative to hand-crafted features. For each agent, we encode its SDC-normalized trajectory [91 timesteps × 4 channels (x, y, vx, vy)] → 768-dimensional embedding, then mean-pool across agents to obtain a scene-level embedding.

MOMENT-1 embeddings did not improve over hand-crafted features for clustering or retrieval on this dataset. This is likely because MOMENT is a generic time series model, not trained on driving data. A domain-specific model like Forecast-MAE (Cheng et al., ICCV 2023) — which uses masked trajectory reconstruction as a pretraining objective — would be a stronger candidate for future work.

## 7. Retrieval System

We build a nearest-neighbor retrieval system in the 22-feature space using Mahalanobis distance, which accounts for feature correlations via the empirical inverse covariance matrix.

Given a query scenario, the system returns the most similar scenarios by feature profile. We evaluate retrieval quality using the CausalAgents labels as ground truth.

**Results:** We select the 10 highest-interestingness scenarios (that have causal labels) as queries, retrieve their 20 nearest neighbors each, and check how many neighbors also contain human-labeled causal agents. **98.6%** do — meaning our feature-space proximity corresponds to real interaction similarity, not just statistical closeness. Precision remains above 89% even at top-200. Note: the `validation_interactive` split has a high baseline interaction rate, so the more discriminative signal is the per-cluster analysis in Section 9, where causal agent counts vary from 1.29 to 9.83 across components.

## 8. Bayesian Rare Event Search

The core practical question: given a large dataset, how do you find the rare/dangerous scenarios without evaluating all of them?

We define an **interestingness score** combining statistical rarity (negative GMM log-likelihood) with safety-critical features (hard braking, low TTC, high deceleration, close encounters). Then we use Bayesian optimization to efficiently search for high-scoring scenarios.

**Method:**
1. Reduce the 22 features to 8 PCA components (74% variance retained) for GP tractability
2. Seed with 100 randomly evaluated scenarios
3. Fit a Gaussian Process (Matern-2.5 kernel) on observed (features, interestingness) pairs
4. Select the next batch of 5 scenarios using Upper Confidence Bound (UCB) acquisition: μ + κσ, balancing exploitation (high predicted score) with exploration (high uncertainty)
5. Repeat for 50 iterations (350 total evaluations = 2.4% of dataset)

**Results:**

| Metric | Bayesian Search | Random Sampling | Improvement |
|--------|----------------|-----------------|-------------|
| Top-50 recall | **72%** | 2% | **36x** |
| Top-100 recall | **73%** | 5% | **15x** |
| Scenarios evaluated | 350 (2.4%) | 350 (2.4%) | — |

The GP surrogate learns the structure of the interestingness landscape and focuses evaluation on the most promising regions of feature space, dramatically outperforming random sampling.

## 9. Ground-Truth Validation with Causal Agent Labels

We validate our unsupervised findings against 4,760 human-labeled scenarios from the CausalAgents benchmark. For each scenario, 5 annotators independently marked which agents causally influenced the SDC. We use majority vote (≥3/5 annotators) for consensus.

**Key finding by GMM component:**

| Component | Mean Causal Agents | vs Overall (3.56) |
|-----------|-------------------|-------------------|
| 7 (dense encounters) | **9.83** | +6.27 |
| 1 | 4.40 | +0.84 |
| 3 | 4.41 | +0.85 |
| 8 (hard braking) | 3.49 | -0.06 |
| 11 (flagged evasive) | **1.29** | **-2.27** |

Component 7 (dense conflict zones) has 2.8x the average causal agent count — these are genuinely complex interactive scenarios. Component 11 (which we flagged as false positives via BEV inspection) has the **lowest** causal count of all components, independently confirming they are not genuinely dangerous.

Interestingly, component 8 (hard braking) is near the overall average for causal agents. Hard braking alone does not imply rich interaction — it may reflect the SDC's conservative behavior rather than a genuinely dangerous situation.

## 10. Conclusions

1. **Hand-crafted physics-based features capture meaningful structure** in driving scenarios. GMM clustering with 22 features identifies interpretable rare-event types that align with human judgment.

2. **Visual inspection is essential.** Scalar summary statistics produce false positives — scenarios flagged as "dangerous" that are actually benign. BEV visualization caught what statistics missed.

3. **Bayesian search makes rare event discovery practical at scale.** A GP-UCB approach finds 72% of the rarest scenarios after evaluating only 2.4% of the data — a 36x improvement over random sampling.

4. **Causal agent labels validate the approach.** The clusters with the highest human-labeled interaction density are exactly those our unsupervised method flags as interesting. The false positives identified visually are confirmed by the lowest causal agent counts.

5. **Learned embeddings remain an open direction.** Generic time series models (MOMENT-1) don't improve over hand-crafted features, but domain-specific masked trajectory models (Forecast-MAE) could capture temporal patterns that scalar features miss — particularly the noise-vs-real-event distinction that causes false positives.

## References

- Waymo Open Motion Dataset. https://waymo.com/open/data/motion/
- R. Roelofs et al., "CausalAgents: A Robustness Benchmark for Motion Forecasting using Causal Relationships," 2022. arXiv:2207.03586
- J. Cheng et al., "Forecast-MAE: Self-supervised Pre-training for Motion Forecasting with Masked Autoencoders," ICCV 2023. arXiv:2308.09882
- M. Zhang et al., "Large Scale Autonomous Driving Scenarios Clustering with Self-supervised Feature Extraction," 2021. arXiv:2103.16101
- WOMD-Reasoning: Y. Li et al., "WOMD-Reasoning: A Large-Scale Language Dataset for Interaction Reasoning in AD," ICML 2025. arXiv:2407.04281
