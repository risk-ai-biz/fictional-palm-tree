""" clustered_optimizer.py

A module for dynamic clustering of categorical features and their interactions, with option to disable clustering completely, and integration with optimizer wrappers (SGD, Adam). """

import numpy as np from scipy.cluster.hierarchy import linkage, fcluster from scipy.spatial.distance import squareform from typing import Dict, Tuple, List, Any

-- Clustering utilities -----------------------------------------------------

def cluster_feature(values: np.ndarray, variances: np.ndarray, min_gap: float, max_clusters: int = None) -> np.ndarray: """ Hierarchical clustering with variance-aware distance.

Parameters
----------
values : np.ndarray, shape (N,)
    Estimates (weights or gradient means).
variances : np.ndarray, shape (N,)
    Variances of each estimate.
min_gap : float
    Minimum merge distance threshold.
max_clusters : int, optional
    Maximum allowed clusters.

Returns
-------
labels : np.ndarray, shape (N,)
    Cluster labels 0..K-1.
"""
diff = np.abs(values[:, None] - values[None, :])
dist = diff / np.sqrt(variances[:, None] + variances[None, :] + 1e-12)
condensed = squareform(dist, checks=False)
Z = linkage(condensed, method='ward')
merge_heights = Z[:, 2]
k = np.searchsorted(merge_heights, min_gap) + 1
if max_clusters is not None:
    k = min(k, max_clusters)
labels = fcluster(Z, k, criterion='maxclust') - 1
return labels.astype(np.int32)

def split_clusters(values: np.ndarray, variances: np.ndarray, labels: np.ndarray, split_thresh: float) -> np.ndarray: """ Split clusters whose RMS dispersion exceeds split_thresh.

Parameters
----------
values : np.ndarray, shape (N,)
variances : np.ndarray, shape (N,)
labels : np.ndarray, shape (N,)
split_thresh : float

Returns
-------
new_labels : np.ndarray, shape (N,)
"""
new_labels = labels.copy()
next_id = new_labels.max() + 1
for c in np.unique(labels):
    idx = np.where(labels == c)[0]
    if idx.size <= 1:
        continue
    mu_c = np.mean(values[idx])
    disp = np.sqrt(np.mean((values[idx] - mu_c)**2))
    if disp > split_thresh:
        sub_vals = values[idx]
        sub_vars = variances[idx]
        sub_labels = cluster_feature(sub_vals, sub_vars, min_gap=0.0, max_clusters=2)
        for loc, sl in zip(idx, sub_labels):
            new_labels[loc] = next_id if sl == 1 else c
        next_id += 1
# rebase labels to 0..K_new-1
uniq = np.unique(new_labels)
remap = {old: i for i, old in enumerate(uniq)}
return np.array([remap[l] for l in new_labels], dtype=np.int32)

def refresh_maps(value_stats: Dict[Any, Tuple[np.ndarray, np.ndarray]], raw_sizes: Dict[Any, int], clusterable: Dict[Any, bool], interaction_pairs: List[Tuple[Any, Any]], min_gap_single: float, split_thresh_single: float, min_gap_pair: float, split_thresh_pair: float) -> Dict[Any, np.ndarray]: """ Build or refresh cluster maps for single and pairwise features. """ cluster_maps: Dict[Any, np.ndarray] = {} # single features for feat, size in raw_sizes.items(): means, vars_ = value_stats[(feat,)] if clusterable.get(feat, True): labels = cluster_feature(means, vars_, min_gap_single) if split_thresh_single > 0: labels = split_clusters(means, vars_, labels, split_thresh_single) else: labels = np.arange(size, dtype=np.int32) cluster_maps[(feat,)] = labels # pairwise interactions for feat_a, feat_b in interaction_pairs: size_a = raw_sizes[feat_a] size_b = raw_sizes[feat_b] means_pair, vars_pair = value_stats[(feat_a, feat_b)] if clusterable.get(feat_a, True) and clusterable.get(feat_b, True): labels = cluster_feature(means_pair, vars_pair, min_gap_pair) if split_thresh_pair > 0: labels = split_clusters(means_pair, vars_pair, labels, split_thresh_pair) elif clusterable.get(feat_a, True) and not clusterable.get(feat_b, True): cmap_a = cluster_maps[(feat_a,)] labels = np.repeat(cmap_a, size_b) * size_b + np.tile(np.arange(size_b), size_a) elif not clusterable.get(feat_a, True) and clusterable.get(feat_b, True): cmap_b = cluster_maps[(feat_b,)] labels = np.repeat(np.arange(size_a), size_b) * (cmap_b.max() + 1) + np.tile(cmap_b, size_a) else: labels = np.arange(size_a * size_b, dtype=np.int32) cluster_maps[(feat_a, feat_b)] = labels.astype(np.int32) return cluster_maps

class ClusteredOptimizer: """ Wrapper for base optimizer with optional clustering.

If cluster_enabled=False, acts as a pass-through SGD/Adam on raw categories.
"""
def __init__(self,
             base_optimizer: Any,
             raw_sizes: Dict[Any, int],
             initial_cluster_maps: Dict[Any, np.ndarray],
             clusterable: Dict[Any, bool],
             interaction_pairs: List[Tuple[Any, Any]],
             min_gap_single: float,
             split_thresh_single: float,
             min_gap_pair: float,
             split_thresh_pair: float,
             refresh_freq: int = 1,
             cluster_enabled: bool = True):
    self.base_optimizer = base_optimizer
    self.raw_sizes = raw_sizes
    self.clusterable = clusterable
    self.interaction_pairs = interaction_pairs
    self.min_gap_single = min_gap_single
    self.split_thresh_single = split_thresh_single
    self.min_gap_pair = min_gap_pair
    self.split_thresh_pair = split_thresh_pair
    self.refresh_freq = refresh_freq
    self.cluster_enabled = cluster_enabled
    # initialize cluster maps: identity if disabled
    if not cluster_enabled:
        self.cluster_maps = self._identity_maps()
        self.refresh_freq = np.iinfo(np.int32).max
    else:
        self.cluster_maps = initial_cluster_maps
    self.step_count = 0
    self._build_views()
    # allocate parameter & gradient arrays
    self.theta_flat = np.zeros(self.total_clusters, dtype=np.float64)
    self.grad_flat = np.zeros_like(self.theta_flat)
    # optimizer state (Adam)
    if hasattr(self.base_optimizer, 'm'):
        self.base_optimizer.m = np.zeros_like(self.theta_flat)
        self.base_optimizer.v = np.zeros_like(self.theta_flat)

def _identity_maps(self) -> Dict[Any, np.ndarray]:
    """Generate identity cluster maps (one-to-one) for raw_sizes."""
    maps: Dict[Any, np.ndarray] = {}
    for feat, size in self.raw_sizes.items():
        maps[(feat,)] = np.arange(size, dtype=np.int32)
    for feat_a, feat_b in self.interaction_pairs:
        size_a = self.raw_sizes[feat_a]
        size_b = self.raw_sizes[feat_b]
        maps[(feat_a, feat_b)] = np.arange(size_a * size_b, dtype=np.int32)
    return maps

def _build_views(self):
    """Compute offsets and total_clusters based on cluster_maps."""
    self.map_keys = list(self.cluster_maps.keys())
    offsets = []
    cursor = 0
    for key in self.map_keys:
        offsets.append(cursor)
        cursor += int(self.cluster_maps[key].max()) + 1
    self.offsets = np.array(offsets, dtype=np.int32)
    self.total_clusters = cursor

def _scatter_params(self, raw_params: Dict[Any, np.ndarray]):
    """Initialize flat parameters from raw per-category arrays."""
    for idx, key in enumerate(self.map_keys):
        cmap = self.cluster_maps[key]
        start = self.offsets[idx]
        seg = np.zeros(int(cmap.max()) + 1, dtype=np.float64)
        counts = np.zeros_like(seg, dtype=int)
        vals = raw_params[key]
        for raw_id, c_id in enumerate(cmap):
            seg[c_id] += vals[raw_id]
            counts[c_id] += 1
        self.theta_flat[start:start+len(seg)] = seg / np.maximum(counts, 1)
    if hasattr(self.base_optimizer, 'm'):
        self.base_optimizer.m[:] = 0
        self.base_optimizer.v[:] = 0

def _repack_state(self, new_maps: Dict[Any, np.ndarray]):
    """Merge old parameter state into new cluster structure."""
    old_maps = self.cluster_maps
    old_theta = self.theta_flat.copy()
    old_m = getattr(self.base_optimizer, 'm', None)
    old_v = getattr(self.base_optimizer, 'v', None)
    old_keys = self.map_keys[:]
    old_offsets = self.offsets.copy()
    # update maps and views
    self.cluster_maps = new_maps
    self._build_views()
    new_theta = np.zeros_like(self.theta_flat)
    if old_m is not None:
        new_m = np.zeros_like(new_theta)
        new_v = np.zeros_like(new_theta)
    # remap each raw index
    for idx, key in enumerate(self.map_keys):
        raw_size = (self.raw_sizes[key[0]] if len(key)==1
                    else self.raw_sizes[key[0]]*self.raw_sizes[key[1]])
        cmap_old = old_maps[key]
        cmap_new = new_maps[key]
        start_old = old_offsets[old_keys.index(key)]
        start_new = self.offsets[idx]
        K_new = int(cmap_new.max())+1
        sums = np.zeros(K_new)
        counts = np.zeros(K_new, dtype=int)
        if old_m is not None:
            m_sums = np.zeros(K_new)
            v_sums = np.zeros(K_new)
        for raw_id in range(raw_size):
            c_old = cmap_old[raw_id]
            i_old = start_old + c_old
            c_new = cmap_new[raw_id]
            sums[c_new] += old_theta[i_old]
            counts[c_new] += 1
            if old_m is not None:
                m_sums[c_new] += old_m[i_old]
                v_sums[c_new] += old_v[i_old]
        new_theta[start_new:start_new+K_new] = sums/np.maximum(counts,1)
        if old_m is not None:
            new_m[start_new:start_new+K_new] = m_sums/np.maximum(counts,1)
            new_v[start_new:start_new+K_new] = v_sums/np.maximum(counts,1)
    self.theta_flat = new_theta
    if old_m is not None:
        self.base_optimizer.m = new_m
        self.base_optimizer.v = new_v

def step(self, raw_grads: Dict[Any, np.ndarray],
         value_stats: Dict[Any, Tuple[np.ndarray, np.ndarray]]):
    """Perform one optimization step; apply clustering if enabled."""
    if not self.cluster_enabled:
        # pass-through: scatter raw grads directly and update
        flat_grads = np.concatenate([raw_grads[key].ravel() for key in self.map_keys])
        self.base_optimizer.step(flat_grads, flat_grads)
        return

    # remap raw to cluster grads
    self.grad_flat.fill(0.0)
    for idx, key in enumerate(self.map_keys):
        cmap = self.cluster_maps[key]
        start = self.offsets[idx]
        for raw_id, c_id in enumerate(cmap):
            self.grad_flat[start + c_id] += raw_grads[key][raw_id]
    # delegate update
    self.base_optimizer.step(self.theta_flat, self.grad_flat)
    self.step_count += 1
    # refresh if enabled
    if self.cluster_enabled and self.step_count % self.refresh_freq == 0:
        new_maps = refresh_maps(
            value_stats, self.raw_sizes, self.clusterable,
            self.interaction_pairs,
            self.min_gap_single, self.split_thresh_single,
            self.min_gap_pair, self.split_thresh_pair
        )
        self._repack_state(new_maps)
        self.step_count = 0

This snippet includes a `cluster_enabled` flag in `ClusteredOptimizer`. If set to `False`, clustering is disabled and the optimizer simply concatenates raw gradients and parameters for pass‑through updates. Otherwise, it runs the full cluster‑aware logic. Let me know if you need any further tweaks!

