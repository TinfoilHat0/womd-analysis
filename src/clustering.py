"""
Scenario clustering pipeline.

Preprocessing:
- Drop highly correlated features (|r| > 0.8)
- Binarize zero-inflated counts
- Log-transform skewed features
- Standardize (z-score)

Clustering:
- K-means with elbow method
- HDBSCAN for density-based clusters
- GMM with BIC model selection

Visualization:
- UMAP 2D projection colored by cluster
"""

import json
import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import hdbscan
import umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Load & preprocess
# ---------------------------------------------------------------------------

def load_and_preprocess(path='data/features.json'):
    with open(path) as f:
        raw = json.load(f)

    scenario_ids = [r['scenario_id'] for r in raw]

    # Features to DROP (|r| > 0.8 with a kept feature)
    drop = {'ego_displacement', 'ego_max_speed', 'num_vehicles', 'agent_density'}

    # Features to BINARIZE (zero-inflated)
    binarize = {'hard_brake_count', 'num_cyclists', 'num_speed_bumps'}

    # Features to LOG-TRANSFORM (right-skewed continuous)
    log_transform = {'min_ttc', 'num_pedestrians'}

    # Build feature matrix
    # Determine final feature names in order
    all_features = [k for k in raw[0].keys() if k != 'scenario_id' and k not in drop]

    feature_names = []
    for f in all_features:
        if f in binarize:
            feature_names.append(f'has_{f.replace("num_", "").replace("_count", "")}')
        elif f in log_transform:
            feature_names.append(f'log_{f}')
        else:
            feature_names.append(f)

    n = len(raw)
    X = np.zeros((n, len(feature_names)))

    for i, r in enumerate(raw):
        col = 0
        for f in all_features:
            val = r[f] if r[f] is not None else 0
            if f in binarize:
                X[i, col] = 1 if val > 0 else 0
            elif f in log_transform:
                X[i, col] = math.log1p(val)
            else:
                X[i, col] = val
            col += 1

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, feature_names, scenario_ids, scaler


# ---------------------------------------------------------------------------
# 2. K-means with elbow
# ---------------------------------------------------------------------------

def run_kmeans_elbow(X, k_range=range(2, 16)):
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)
        print(f'  k={k}: inertia={km.inertia_:.0f}')
    return list(k_range), inertias


def plot_elbow(ks, inertias, path='data/elbow.png'):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, inertias, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia (within-cluster sum of squares)')
    ax.set_title('K-Means Elbow Plot')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved elbow plot to {path}')


# ---------------------------------------------------------------------------
# 3. GMM with BIC
# ---------------------------------------------------------------------------

def run_gmm_bic(X, k_range=range(2, 16)):
    bics = []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type='full',
                               n_init=3, random_state=42)
        gmm.fit(X)
        bics.append(gmm.bic(X))
        print(f'  k={k}: BIC={gmm.bic(X):.0f}')
    return list(k_range), bics


def plot_bic(ks, bics, path='data/gmm_bic.png'):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, bics, 'o-', linewidth=2, markersize=6, color='tab:orange')
    ax.set_xlabel('Number of Components (k)')
    ax.set_ylabel('BIC (lower is better)')
    ax.set_title('GMM Model Selection (BIC)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved BIC plot to {path}')


# ---------------------------------------------------------------------------
# 4. HDBSCAN
# ---------------------------------------------------------------------------

def run_hdbscan(X, min_cluster_size=30, min_samples=10):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
    )
    labels = clusterer.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f'  HDBSCAN: {n_clusters} clusters, {n_noise} noise points ({n_noise/len(labels)*100:.1f}%)')
    return labels, clusterer


# ---------------------------------------------------------------------------
# 4. Dimensionality reduction (PCA + UMAP)
# ---------------------------------------------------------------------------

def run_pca(X, feature_names):
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(X)
    print(f'  PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% variance')
    print(f'  PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% variance')
    print(f'  Total: {sum(pca.explained_variance_ratio_)*100:.1f}%')

    # Top loadings per component
    for i, pc_name in enumerate(['PC1', 'PC2']):
        loadings = list(zip(feature_names, pca.components_[i]))
        loadings.sort(key=lambda x: abs(x[1]), reverse=True)
        top = loadings[:5]
        signs = [f'{name} ({val:+.2f})' for name, val in top]
        print(f'  {pc_name} top loadings: {", ".join(signs)}')

    return embedding, pca


def run_umap(X, n_neighbors=30, min_dist=0.3, random_state=42):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(X)
    return embedding


def plot_clusters(embedding, labels, title, path, xlabel='UMAP 1', ylabel='UMAP 2',
                  label_name='Cluster'):
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_labels = sorted(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(unique_labels), 1)))

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c='lightgray', s=8, alpha=0.4, label='Noise')
        else:
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[colors[idx % 20]], s=12, alpha=0.6,
                       label=f'{label_name} {label}')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


# ---------------------------------------------------------------------------
# 5. Cluster profiling
# ---------------------------------------------------------------------------

def profile_clusters(X, feature_names, labels, label_name='Cluster'):
    """Print mean feature values per cluster (in standardized space)."""
    unique_labels = sorted(set(labels))
    print(f'\n{"Feature":<30}', end='')
    for label in unique_labels:
        name = 'Noise' if label == -1 else f'{label_name} {label}'
        print(f'{name:>10}', end='')
    print(f'{"Overall":>10}')
    print('-' * (30 + 10 * (len(unique_labels) + 1)))

    for j, fname in enumerate(feature_names):
        print(f'{fname:<30}', end='')
        for label in unique_labels:
            mask = labels == label
            print(f'{X[mask, j].mean():>10.2f}', end='')
        print(f'{X[:, j].mean():>10.2f}')

    # Cluster sizes
    print(f'\n{"Size":<30}', end='')
    for label in unique_labels:
        print(f'{(labels == label).sum():>10}', end='')
    print(f'{len(labels):>10}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=== Loading & preprocessing ===')
    X, feature_names, scenario_ids, scaler = load_and_preprocess()
    print(f'Feature matrix: {X.shape[0]} scenarios x {X.shape[1]} features')
    print(f'Features: {feature_names}')

    print('\n=== K-Means elbow ===')
    ks, inertias = run_kmeans_elbow(X)
    plot_elbow(ks, inertias)

    # Pick k based on elbow — we'll use k=6 as starting point, adjust after seeing elbow
    chosen_k = 6
    print(f'\n=== K-Means (k={chosen_k}) ===')
    km = KMeans(n_clusters=chosen_k, n_init=10, random_state=42)
    km_labels = km.fit_predict(X)
    for k in range(chosen_k):
        print(f'  Cluster {k}: {(km_labels == k).sum()} scenarios')

    print('\n=== GMM BIC ===')
    gmm_ks, bics = run_gmm_bic(X)
    plot_bic(gmm_ks, bics)

    best_gmm_k = gmm_ks[np.argmin(bics)]
    print(f'\n=== GMM (k={best_gmm_k}) ===')
    gmm = GaussianMixture(n_components=best_gmm_k, covariance_type='full',
                           n_init=3, random_state=42)
    gmm.fit(X)
    gmm_labels = gmm.predict(X)
    gmm_probs = gmm.predict_proba(X)
    gmm_loglik = gmm.score_samples(X)
    for k in range(best_gmm_k):
        print(f'  Component {k}: {(gmm_labels == k).sum()} scenarios')
    print(f'  Log-likelihood range: [{gmm_loglik.min():.1f}, {gmm_loglik.max():.1f}]')
    print(f'  Bottom 1% threshold: {np.percentile(gmm_loglik, 1):.1f}')

    print('\n=== HDBSCAN ===')
    hdb_labels, hdb_clusterer = run_hdbscan(X)

    print('\n=== PCA ===')
    pca_embedding, pca_model = run_pca(X, feature_names)

    print('\n=== UMAP ===')
    umap_embedding = run_umap(X)

    print('\n=== Plotting ===')
    # PCA plots
    plot_clusters(pca_embedding, km_labels, f'K-Means (k={chosen_k}) on PCA',
                  'data/clusters_kmeans_pca.png', xlabel='PC1', ylabel='PC2')
    plot_clusters(pca_embedding, hdb_labels, 'HDBSCAN on PCA',
                  'data/clusters_hdbscan_pca.png', xlabel='PC1', ylabel='PC2', label_name='Cluster')
    plot_clusters(pca_embedding, gmm_labels, f'GMM (k={best_gmm_k}) on PCA',
                  'data/clusters_gmm_pca.png', xlabel='PC1', ylabel='PC2', label_name='Component')
    # UMAP plots
    plot_clusters(umap_embedding, km_labels, f'K-Means (k={chosen_k}) on UMAP',
                  'data/clusters_kmeans.png')
    plot_clusters(umap_embedding, hdb_labels, 'HDBSCAN on UMAP',
                  'data/clusters_hdbscan.png', label_name='Cluster')
    plot_clusters(umap_embedding, gmm_labels, f'GMM (k={best_gmm_k}) on UMAP',
                  'data/clusters_gmm.png', label_name='Component')

    # GMM log-likelihood heatmap on UMAP
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1],
                    c=gmm_loglik, cmap='RdYlBu', s=8, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='Log-likelihood')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('GMM Log-Likelihood on UMAP (low = rare)')
    fig.tight_layout()
    fig.savefig('data/gmm_loglikelihood.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved data/gmm_loglikelihood.png')

    print('\n=== K-Means Cluster Profiles ===')
    profile_clusters(X, feature_names, km_labels)

    print('\n=== HDBSCAN Cluster Profiles ===')
    profile_clusters(X, feature_names, hdb_labels)

    print('\n=== GMM Component Profiles ===')
    profile_clusters(X, feature_names, gmm_labels, label_name='Comp')
