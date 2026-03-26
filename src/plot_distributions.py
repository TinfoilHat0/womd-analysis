"""Generate distribution, correlation, and risk scatter plots from features.json."""

import json
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('data/features.json') as f:
    data = json.load(f)

n = len(data)
skip = {'scenario_id'}
features = [k for k in data[0].keys() if k not in skip]

# 1. Distributions
fig, axes = plt.subplots(5, 6, figsize=(24, 18))
axes = axes.flatten()
for i, feat in enumerate(features):
    vals = [d[feat] for d in data if d[feat] is not None]
    ax = axes[i]
    ax.hist(vals, bins=50, edgecolor='black', linewidth=0.3, alpha=0.7)
    median = sorted(vals)[len(vals)//2]
    ax.axvline(median, color='red', linestyle='--', linewidth=1, label=f'med={median:.2f}')
    ax.set_title(feat, fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
for j in range(len(features), len(axes)):
    axes[j].set_visible(False)
fig.suptitle(f'Feature Distributions (n={n})', fontsize=14)
fig.tight_layout()
fig.savefig('data/feature_distributions.png', dpi=150)
plt.close(fig)
print('Saved feature_distributions.png')

# 2. Correlations
mat = np.zeros((n, len(features)))
for i, d in enumerate(data):
    for j, feat in enumerate(features):
        mat[i, j] = d[feat] if d[feat] is not None else 0
corr = np.corrcoef(mat.T)
fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(features)))
ax.set_yticks(range(len(features)))
ax.set_xticklabels(features, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(features, fontsize=8)
for i in range(len(features)):
    for j in range(len(features)):
        if abs(corr[i, j]) > 0.5:
            ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center', fontsize=6,
                    color='white' if abs(corr[i, j]) > 0.7 else 'black')
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title(f'Feature Correlations (n={n})')
fig.tight_layout()
fig.savefig('data/feature_correlations.png', dpi=150)
plt.close(fig)
print('Saved feature_correlations.png')

# 3. Risk scatter
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ttc = [d['min_ttc'] if d['min_ttc'] is not None else 50 for d in data]
dist = [d['min_distance'] if d['min_distance'] is not None else 50 for d in data]
agents = [d['num_agents'] for d in data]
speed = [d['ego_mean_speed'] for d in data]
decel = [d['ego_max_decel'] for d in data]
lanes = [d['num_lanes'] for d in data]

sc = axes[0].scatter(ttc, dist, c=decel, s=5, alpha=0.4, cmap='YlOrRd')
axes[0].set_xlabel('Min TTC (s)')
axes[0].set_ylabel('Min Distance (m)')
axes[0].set_title('TTC vs Distance (color=max decel)')
axes[0].set_xlim(0, 15)
axes[0].set_ylim(0, 30)
fig.colorbar(sc, ax=axes[0], label='Max Decel')

sc = axes[1].scatter(agents, lanes, c=[math.log1p(t) for t in ttc], s=5, alpha=0.4, cmap='RdYlGn')
axes[1].set_xlabel('Num Agents')
axes[1].set_ylabel('Num Lanes')
axes[1].set_title('Scene Complexity (color=log TTC)')
fig.colorbar(sc, ax=axes[1], label='log(1+TTC)')

sc = axes[2].scatter(speed, decel, c=dist, s=5, alpha=0.4, cmap='viridis_r')
axes[2].set_xlabel('Ego Mean Speed (m/s)')
axes[2].set_ylabel('Max Deceleration (m/s²)')
axes[2].set_title('Speed vs Braking (color=min dist)')
fig.colorbar(sc, ax=axes[2], label='Min Distance')

fig.suptitle(f'Risk Analysis (n={n})', fontsize=13)
fig.tight_layout()
fig.savefig('data/risk_scatter.png', dpi=150)
plt.close(fig)
print('Saved risk_scatter.png')

# 4. Key stats
print(f'\n=== Key Statistics (n={n}) ===')
for feat in features:
    vals = sorted([d[feat] for d in data if d[feat] is not None])
    m = len(vals)
    mean = sum(vals) / m
    print(f'{feat:<30} min={vals[0]:<10.2f} med={vals[m//2]:<10.2f} max={vals[-1]:<10.2f} mean={mean:<10.2f}')
