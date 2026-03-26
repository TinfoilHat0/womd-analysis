"""
Scenario retrieval, Bayesian rare event search, and causal agent evaluation.

1. Retrieval: Mahalanobis-distance nearest neighbors in 22-feature space
2. Bayesian search: GP surrogate with UCB acquisition for efficient rare event discovery
3. Evaluation: Cross-reference with human-labeled causal agent annotations

Usage:
    python src/retrieval.py [--features data/features.json]
                            [--causal data/causal_agents/causal_labels.tfrecord]
"""

import argparse
import math
import struct
import os
import sys

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.clustering import load_and_preprocess


# ---------------------------------------------------------------------------
# 1. Causal label loading
# ---------------------------------------------------------------------------

def _decode_varint(data, pos):
    result, shift = 0, 0
    while True:
        b = data[pos]
        result |= (b & 0x7F) << shift
        pos += 1
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def _decode_field(data, pos):
    if pos >= len(data):
        return None, None, None, pos
    tag, pos = _decode_varint(data, pos)
    field_num = tag >> 3
    wire_type = tag & 0x07
    if wire_type == 2:  # length-delimited
        length, pos = _decode_varint(data, pos)
        return field_num, wire_type, data[pos:pos + length], pos + length
    elif wire_type == 0:  # varint
        value, pos = _decode_varint(data, pos)
        return field_num, wire_type, value, pos
    return field_num, wire_type, None, pos


def load_causal_labels(path):
    """
    Load causal agent labels from TFRecord.

    Returns dict: {scenario_id: {
        'annotator_labels': [list of 5 sets of track_ids],
        'consensus_causal': set of track_ids (>=3/5 annotators agree),
        'num_causal': int (consensus count)
    }}
    """
    from src.feature_extraction import read_tfrecord

    records = read_tfrecord(path)
    labels = {}

    for rec in records:
        pos = 0
        annotators = []
        scenario_id = None

        while pos < len(rec):
            fnum, wtype, val, pos = _decode_field(rec, pos)
            if fnum is None:
                break
            if fnum == 1 and wtype == 2:
                tracks = []
                ipos = 0
                while ipos < len(val):
                    ifnum, iwtype, ival, ipos = _decode_field(val, ipos)
                    if ifnum == 1 and iwtype == 2:
                        tracks.append(ival.decode('utf-8'))
                annotators.append(set(tracks))
            elif fnum == 2 and wtype == 2:
                scenario_id = val.decode('utf-8')

        if scenario_id is None:
            continue

        # Consensus: track labeled by >= 3 of 5 annotators
        all_tracks = set()
        for a in annotators:
            all_tracks |= a
        consensus = set()
        for track_id in all_tracks:
            count = sum(1 for a in annotators if track_id in a)
            if count >= 3:
                consensus.add(track_id)

        labels[scenario_id] = {
            'annotator_labels': annotators,
            'consensus_causal': consensus,
            'num_causal': len(consensus),
        }

    return labels


# ---------------------------------------------------------------------------
# 2. Interestingness scoring
# ---------------------------------------------------------------------------

def compute_interestingness(X, feature_names, gmm, alpha=1.0):
    """
    Composite rarity-danger score.

    interestingness = -gmm_loglik + alpha * danger_score

    danger_score uses z-scored safety features:
        has_hard_brake + close_encounter_count - log_min_ttc + ego_max_decel
    """
    loglik = gmm.score_samples(X)

    # Find safety feature indices
    danger_features = {
        'has_hard_brake': 1.0,
        'close_encounter_count': 1.0,
        'log_min_ttc': -1.0,       # low TTC = dangerous
        'ego_max_decel': 1.0,
    }

    danger_score = np.zeros(len(X))
    for fname, weight in danger_features.items():
        if fname in feature_names:
            idx = feature_names.index(fname)
            danger_score += weight * X[:, idx]

    scores = -loglik + alpha * danger_score
    return scores, loglik, danger_score


# ---------------------------------------------------------------------------
# 3. Scenario retrieval
# ---------------------------------------------------------------------------

def build_retrieval_index(X):
    """Compute inverse covariance matrix for Mahalanobis distance."""
    cov = np.cov(X.T) + 1e-6 * np.eye(X.shape[1])
    VI = np.linalg.inv(cov)
    return VI


def retrieve_similar(query_idx, X, VI, k=10):
    """Return top-K most similar scenarios by Mahalanobis distance."""
    query = X[query_idx:query_idx + 1]
    dists = cdist(query, X, metric='mahalanobis', VI=VI).flatten()
    # Exclude self
    dists[query_idx] = np.inf
    top_k = np.argsort(dists)[:k]
    return top_k, dists[top_k]


def retrieve_by_id(scenario_id, scenario_ids, X, VI, k=10):
    """Retrieve by scenario ID string."""
    idx = scenario_ids.index(scenario_id)
    top_k, dists = retrieve_similar(idx, X, VI, k)
    return [scenario_ids[i] for i in top_k], dists


# ---------------------------------------------------------------------------
# 4. Bayesian rare event search
# ---------------------------------------------------------------------------

def bayesian_search(X, scores, n_init=100, n_iter=200, batch_size=5,
                    n_pca=8, kappa=2.0, random_state=42):
    """
    Bayesian optimization to find high-interestingness scenarios.

    Uses GP surrogate on PCA-reduced features with UCB acquisition.
    Returns search history and comparison with random baseline.
    """
    rng = np.random.RandomState(random_state)

    # PCA reduction for GP tractability
    pca = PCA(n_components=n_pca)
    X_pca = pca.fit_transform(X)
    pca_var = pca.explained_variance_ratio_.sum()
    print(f'  GP uses {n_pca} PCA components ({pca_var * 100:.1f}% variance)')

    n = len(X)

    # Seed with random initial observations
    observed = set(rng.choice(n, size=n_init, replace=False))
    history = {
        'n_observed': [n_init],
        'top50_recall': [],
        'top100_recall': [],
        'max_score_found': [],
    }

    # Ground truth top-K
    true_top50 = set(np.argsort(scores)[-50:])
    true_top100 = set(np.argsort(scores)[-100:])

    def eval_recall():
        found_top50 = len(observed & true_top50) / 50
        found_top100 = len(observed & true_top100) / 100
        max_score = scores[list(observed)].max()
        history['top50_recall'].append(found_top50)
        history['top100_recall'].append(found_top100)
        history['max_score_found'].append(max_score)

    eval_recall()

    kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-3)

    for iteration in range(n_iter):
        obs_list = sorted(observed)
        X_obs = X_pca[obs_list]
        y_obs = scores[obs_list]

        # Fit GP
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                       random_state=random_state, normalize_y=True)
        gp.fit(X_obs, y_obs)

        # Predict on unobserved
        unobs = np.array(sorted(set(range(n)) - observed))
        mu, sigma = gp.predict(X_pca[unobs], return_std=True)

        # UCB acquisition
        acq = mu + kappa * sigma
        top_batch = unobs[np.argsort(acq)[-batch_size:]]

        observed.update(top_batch)
        history['n_observed'].append(len(observed))
        eval_recall()

        if (iteration + 1) % 50 == 0:
            print(f'  Iter {iteration + 1}/{n_iter}: observed={len(observed)}, '
                  f'top50 recall={history["top50_recall"][-1]:.0%}, '
                  f'top100 recall={history["top100_recall"][-1]:.0%}')

    # Random baseline comparison
    random_observed = set(rng.choice(n, size=len(observed), replace=False))
    history['random_top50_recall'] = len(random_observed & true_top50) / 50
    history['random_top100_recall'] = len(random_observed & true_top100) / 100
    history['total_observed'] = len(observed)
    history['observed_indices'] = sorted(observed)

    return history


# ---------------------------------------------------------------------------
# 5. Evaluation against causal labels
# ---------------------------------------------------------------------------

def evaluate_clusters_causal(scenario_ids, gmm_labels, causal_labels):
    """Check whether GMM danger clusters have more causal agents."""
    overlap_ids = [sid for sid in scenario_ids if sid in causal_labels]
    print(f'  Overlap: {len(overlap_ids)} of {len(scenario_ids)} scenarios '
          f'have causal labels ({len(overlap_ids) / len(scenario_ids) * 100:.0f}%)')

    # Per-component mean causal agent count
    component_causal = {}
    for sid in overlap_ids:
        idx = scenario_ids.index(sid)
        comp = gmm_labels[idx]
        nc = causal_labels[sid]['num_causal']
        component_causal.setdefault(comp, []).append(nc)

    overall_mean = np.mean([causal_labels[sid]['num_causal'] for sid in overlap_ids])

    print(f'\n  {"Component":<12} {"Count":<8} {"Mean Causal":<14} {"vs Overall":<12}')
    print(f'  {"-" * 46}')
    for comp in sorted(component_causal.keys()):
        vals = component_causal[comp]
        mean_c = np.mean(vals)
        diff = mean_c - overall_mean
        marker = ' ***' if diff > 0.5 else ''
        print(f'  {comp:<12} {len(vals):<8} {mean_c:<14.2f} {diff:+.2f}{marker}')
    print(f'  {"Overall":<12} {len(overlap_ids):<8} {overall_mean:<14.2f}')

    return component_causal, overall_mean


def evaluate_retrieval_causal(X, scenario_ids, causal_labels, VI, scores, n_queries=10):
    """
    For the top-N most interesting scenarios with causal labels,
    retrieve neighbors and check if they also have high causal counts.
    """
    overlap = [(scenario_ids.index(sid), sid) for sid in scenario_ids
               if sid in causal_labels]
    if not overlap:
        print('  No overlap with causal labels!')
        return

    # Pick top queries by interestingness among overlapping scenarios
    overlap_scores = [(idx, sid, scores[idx]) for idx, sid in overlap]
    overlap_scores.sort(key=lambda x: x[2], reverse=True)
    queries = overlap_scores[:n_queries]

    print(f'\n  {"Query ID":<20} {"Score":<8} {"Causal":<8} '
          f'{"Nbr Mean Causal":<18} {"Nbr in Overlap":<16}')
    print(f'  {"-" * 70}')

    all_precision = []
    for q_idx, q_sid, q_score in queries:
        q_causal = causal_labels[q_sid]['num_causal']
        top_k, dists = retrieve_similar(q_idx, X, VI, k=20)

        # How many neighbors have causal labels and what's their mean causal count?
        nbr_causal = []
        for nbr_idx in top_k:
            nbr_sid = scenario_ids[nbr_idx]
            if nbr_sid in causal_labels:
                nbr_causal.append(causal_labels[nbr_sid]['num_causal'])

        mean_nbr = np.mean(nbr_causal) if nbr_causal else 0
        n_overlap = len(nbr_causal)
        precision = sum(1 for c in nbr_causal if c > 0) / max(len(nbr_causal), 1)
        all_precision.append(precision)

        print(f'  {q_sid:<20} {q_score:<8.1f} {q_causal:<8} '
              f'{mean_nbr:<18.2f} {n_overlap:<16}')

    mean_prec = np.mean(all_precision)
    print(f'\n  Mean precision@20 (fraction of neighbors with causal agents): {mean_prec:.2%}')
    return mean_prec


def precision_at_k(scores, scenario_ids, causal_labels, k=100):
    """Among top-K by interestingness, what fraction have >= 1 causal agent?"""
    top_k_idx = np.argsort(scores)[-k:]
    hits = 0
    evaluated = 0
    for idx in top_k_idx:
        sid = scenario_ids[idx]
        if sid in causal_labels:
            evaluated += 1
            if causal_labels[sid]['num_causal'] > 0:
                hits += 1
    prec = hits / max(evaluated, 1)
    print(f'  Precision@{k}: {hits}/{evaluated} = {prec:.2%} '
          f'(top-{k} by interestingness that have causal labels, fraction with causal agents)')
    return prec


# ---------------------------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------------------------

def plot_bayesian_convergence(history, path='data/bayesian_convergence.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Recall curves
    ax1.plot(history['n_observed'], history['top50_recall'],
             'o-', markersize=3, label='Top-50 recall (Bayesian)')
    ax1.plot(history['n_observed'], history['top100_recall'],
             's-', markersize=3, label='Top-100 recall (Bayesian)')
    ax1.axhline(history['random_top50_recall'], color='red', linestyle='--',
                alpha=0.7, label=f'Random top-50 recall ({history["random_top50_recall"]:.0%})')
    ax1.axhline(history['random_top100_recall'], color='orange', linestyle='--',
                alpha=0.7, label=f'Random top-100 recall ({history["random_top100_recall"]:.0%})')
    ax1.set_xlabel('Scenarios Evaluated')
    ax1.set_ylabel('Recall')
    ax1.set_title('Bayesian Search: Recall of True Top-K')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Max score found
    ax2.plot(history['n_observed'], history['max_score_found'], 'g-', linewidth=2)
    ax2.set_xlabel('Scenarios Evaluated')
    ax2.set_ylabel('Max Interestingness Score Found')
    ax2.set_title('Bayesian Search: Best Score Over Time')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


def plot_causal_by_cluster(component_causal, overall_mean, path='data/causal_by_cluster.png'):
    comps = sorted(component_causal.keys())
    means = [np.mean(component_causal[c]) for c in comps]
    sizes = [len(component_causal[c]) for c in comps]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(comps)), means, color='steelblue', alpha=0.8)
    ax.axhline(overall_mean, color='red', linestyle='--', linewidth=2,
               label=f'Overall mean ({overall_mean:.2f})')

    # Color bars above mean
    for i, (bar, m) in enumerate(zip(bars, means)):
        if m > overall_mean + 0.5:
            bar.set_color('coral')

    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels([f'C{c}\n(n={s})' for c, s in zip(comps, sizes)], fontsize=8)
    ax.set_xlabel('GMM Component')
    ax.set_ylabel('Mean Causal Agent Count')
    ax.set_title('Causal Agent Count by GMM Component')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


def plot_interestingness_umap(X, scores, path='data/interestingness_umap.png'):
    """UMAP colored by interestingness score."""
    import umap
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
    embedding = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                    c=scores, cmap='hot', s=8, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='Interestingness Score')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Interestingness Score on UMAP (high = rare + dangerous)')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', default='data/features.json')
    parser.add_argument('--causal', default='data/causal_agents/causal_labels.tfrecord')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight of danger score vs rarity')
    parser.add_argument('--n-init', type=int, default=100,
                        help='Initial random sample for Bayesian search')
    parser.add_argument('--n-iter', type=int, default=200,
                        help='Bayesian search iterations')
    args = parser.parse_args()

    # --- Load & prep ---
    print('=== Loading features ===')
    X, feature_names, scenario_ids, scaler = load_and_preprocess(args.features)
    print(f'  {X.shape[0]} scenarios, {X.shape[1]} features')

    # --- Fit GMM ---
    print('\n=== Fitting GMM (k=14) ===')
    gmm = GaussianMixture(n_components=14, covariance_type='full',
                           n_init=3, random_state=42)
    gmm.fit(X)
    gmm_labels = gmm.predict(X)

    # --- Interestingness ---
    print('\n=== Computing interestingness scores ===')
    scores, loglik, danger = compute_interestingness(X, feature_names, gmm, alpha=args.alpha)
    print(f'  Score range: [{scores.min():.1f}, {scores.max():.1f}]')
    print(f'  Top 5 most interesting:')
    top5 = np.argsort(scores)[-5:][::-1]
    for idx in top5:
        print(f'    {scenario_ids[idx]}: score={scores[idx]:.1f} '
              f'(loglik={loglik[idx]:.1f}, danger={danger[idx]:.1f}, comp={gmm_labels[idx]})')

    # --- Retrieval ---
    print('\n=== Building retrieval index ===')
    VI = build_retrieval_index(X)

    print('\n=== Retrieval demo ===')
    # Use the most interesting scenario as query
    query_idx = np.argmax(scores)
    query_id = scenario_ids[query_idx]
    print(f'  Query: {query_id} (score={scores[query_idx]:.1f})')
    top_k, dists = retrieve_similar(query_idx, X, VI, k=10)
    print(f'  Top-10 neighbors:')
    for rank, (nbr_idx, d) in enumerate(zip(top_k, dists)):
        print(f'    {rank + 1}. {scenario_ids[nbr_idx]} '
              f'(dist={d:.2f}, score={scores[nbr_idx]:.1f}, comp={gmm_labels[nbr_idx]})')

    # --- Load causal labels ---
    print('\n=== Loading causal agent labels ===')
    if os.path.exists(args.causal):
        causal = load_causal_labels(args.causal)
        print(f'  Loaded {len(causal)} labeled scenarios')

        # --- Evaluate clusters ---
        print('\n=== Cluster vs Causal Agent Evaluation ===')
        comp_causal, overall_mean = evaluate_clusters_causal(
            scenario_ids, gmm_labels, causal)
        plot_causal_by_cluster(comp_causal, overall_mean)

        # --- Evaluate retrieval ---
        print('\n=== Retrieval vs Causal Agent Evaluation ===')
        evaluate_retrieval_causal(X, scenario_ids, causal, VI, scores)

        # --- Precision@K ---
        print('\n=== Precision@K ===')
        for k in [50, 100, 200]:
            precision_at_k(scores, scenario_ids, causal, k=k)
    else:
        print(f'  Causal labels not found at {args.causal}, skipping evaluation')

    # --- Bayesian search ---
    print(f'\n=== Bayesian Rare Event Search (n_init={args.n_init}, n_iter={args.n_iter}) ===')
    history = bayesian_search(X, scores, n_init=args.n_init, n_iter=args.n_iter)
    print(f'\n  Final: observed {history["total_observed"]} scenarios')
    print(f'  Bayesian top-50 recall:  {history["top50_recall"][-1]:.0%}')
    print(f'  Random top-50 recall:    {history["random_top50_recall"]:.0%}')
    print(f'  Bayesian top-100 recall: {history["top100_recall"][-1]:.0%}')
    print(f'  Random top-100 recall:   {history["random_top100_recall"]:.0%}')
    plot_bayesian_convergence(history)

    # --- UMAP interestingness ---
    print('\n=== Plotting interestingness UMAP ===')
    plot_interestingness_umap(X, scores)

    print('\nDone.')
