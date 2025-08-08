
"""
clustered_optimizer.py
======================

A clean, typed, and production-oriented module for **clustered additive models**
with optional **pairwise interactions**, designed for large, distributed SGD/Adam pipelines.

This file synthesizes and simplifies the design:
- raw_id → local_cluster → global_slot lookups
- pairwise interactions via clustering flattened raw pairs
- constraints for un/clusterable features
- optional null cluster for weak categories
- explicit splitting of over-dispersed clusters
- ClusteredOptimizer wrapper to remap grads, update params, and refresh maps

Author: you + assistant
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Protocol, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


# ----------------------------
# Type aliases
# ----------------------------

Feature = str
Key = Tuple[Feature, ...]
ArrayF64 = np.ndarray
ArrayI32 = np.ndarray

ValueStats = Mapping[Key, Tuple[ArrayF64, ArrayF64]]
ClusterMaps = Dict[Key, ArrayI32]


# ----------------------------
# Optimizer protocol
# ----------------------------

class Optimizer(Protocol):
    def step(self, params: ArrayF64, grads: ArrayF64) -> None:
        """Update params in-place given grads."""


# ----------------------------
# Schema & Clustering configuration
# ----------------------------

@dataclass(frozen=True)
class ModelSchema:
    raw_sizes: Mapping[Feature, int]
    pairs: List[Tuple[Feature, Feature]] = field(default_factory=list)

    def raw_size_for(self, key: Key) -> int:
        if len(key) == 1:
            return self.raw_sizes[key[0]]
        if len(key) == 2:
            return self.raw_sizes[key[0]] * self.raw_sizes[key[1]]
        raise ValueError("Only single features '(f,)' or pairs '(f,g)' are supported.")

    def ordered_keys(self) -> List[Key]:
        singles = [(f,) for f in self.raw_sizes.keys()]
        return singles + [tuple(p) for p in self.pairs]


@dataclass(frozen=True)
class ClusteringConfig:
    min_gap_single: float = 1.0
    split_thresh_single: float = 0.0
    min_gap_pair: float = 1.0
    split_thresh_pair: float = 0.0

    clusterable: Mapping[Feature, bool] = field(default_factory=dict)

    null_mu: Optional[float] = None
    null_var: Optional[float] = None

    cluster_enabled: bool = True


# ----------------------------
# Reducer helper
# ----------------------------

def gradient_stats_from_accumulators(
    sum_g: ArrayF64, sum_g2: ArrayF64, count: ArrayF64
) -> Tuple[ArrayF64, ArrayF64]:
    safe = np.maximum(count, 1.0)
    mean = sum_g / safe
    var = sum_g2 / safe - mean**2
    var = np.maximum(var, 0.0)
    return mean, var


# ----------------------------
# Clustering utilities
# ----------------------------

def _ward_labels(
    values: ArrayF64, variances: ArrayF64, min_gap: float, max_clusters: Optional[int] = None
) -> ArrayI32:
    n = int(values.size)
    if n <= 1:
        return np.zeros(n, dtype=np.int32)
    diff = np.abs(values[:, None] - values[None, :])
    dist = diff / np.sqrt(variances[:, None] + variances[None, :] + 1e-12)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")
    merge_heights = Z[:, 2]
    k = int(np.searchsorted(merge_heights, min_gap) + 1)
    if max_clusters is not None:
        k = min(k, max_clusters)
    labels = fcluster(Z, k, criterion="maxclust") - 1
    return labels.astype(np.int32)


def _apply_null_pool(
    values: ArrayF64, variances: ArrayF64, labels: ArrayI32, null_mu: Optional[float], null_var: Optional[float]
) -> ArrayI32:
    if null_mu is None or null_var is None:
        return labels
    is_null = (np.abs(values) < null_mu) & (variances < null_var)
    if not np.any(is_null):
        return labels
    out = np.empty_like(labels)
    out[is_null] = 0
    non_null = ~is_null
    if np.any(non_null):
        uniq = np.unique(labels[non_null])
        remap = {old: i + 1 for i, old in enumerate(uniq)}  # +1 to keep 0 for null
        out[non_null] = np.array([remap[l] for l in labels[non_null]], dtype=np.int32)
    return out


def split_overdispersed(
    values: ArrayF64, variances: ArrayF64, labels: ArrayI32, split_thresh: float
) -> ArrayI32:
    new_labels = labels.copy()
    next_label = int(new_labels.max()) + 1
    uniq = np.unique(labels)
    for c in uniq:
        if c == 0 and np.any(labels == 0):
            continue
        idx = np.where(labels == c)[0]
        if idx.size <= 1:
            continue
        mu = float(np.mean(values[idx]))
        disp = float(np.sqrt(np.mean((values[idx] - mu) ** 2)))
        if disp <= split_thresh:
            continue
        sub = _ward_labels(values[idx], variances[idx], min_gap=0.0, max_clusters=2)
        # Larger child stays with original label c
        keep = 0 if (sub == 0).sum() >= (sub == 1).sum() else 1
        for pos, lab in zip(idx, sub):
            new_labels[pos] = c if lab == keep else next_label
        next_label += 1
    uniq2 = np.unique(new_labels)
    if 0 in uniq2:
        order = [0] + [u for u in uniq2 if u != 0]
    else:
        order = list(uniq2)
    remap = {old: i for i, old in enumerate(order)}
    return np.array([remap[l] for l in new_labels], dtype=np.int32)


def cluster_single(values: ArrayF64, variances: ArrayF64, cfg: ClusteringConfig) -> ArrayI32:
    labels = _ward_labels(values, variances, min_gap=cfg.min_gap_single)
    labels = _apply_null_pool(values, variances, labels, cfg.null_mu, cfg.null_var)
    if cfg.split_thresh_single > 0.0:
        labels = split_overdispersed(values, variances, labels, cfg.split_thresh_single)
    return labels


def cluster_pair(values_flat: ArrayF64, variances_flat: ArrayF64, cfg: ClusteringConfig) -> ArrayI32:
    labels = _ward_labels(values_flat, variances_flat, min_gap=cfg.min_gap_pair)
    if cfg.split_thresh_pair > 0.0:
        labels = split_overdispersed(values_flat, variances_flat, labels, cfg.split_thresh_pair)
    return labels


# ----------------------------
# Map building / refreshing
# ----------------------------

def identity_maps(schema: ModelSchema) -> ClusterMaps:
    maps: ClusterMaps = {}
    for key in schema.ordered_keys():
        maps[key] = np.arange(schema.raw_size_for(key), dtype=np.int32)
    return maps


def refresh_maps(schema: ModelSchema, stats: ValueStats, cfg: ClusteringConfig) -> ClusterMaps:
    if not cfg.cluster_enabled:
        return identity_maps(schema)
    maps: ClusterMaps = {}
    # Singles
    for key in schema.ordered_keys():
        if len(key) != 1:
            continue
        f = key[0]
        mean_f, var_f = stats[(f,)]
        if not cfg.clusterable.get(f, True):
            maps[(f,)] = np.arange(schema.raw_sizes[f], dtype=np.int32)
        else:
            maps[(f,)] = cluster_single(mean_f, var_f, cfg)
    # Pairs
    for key in schema.ordered_keys():
        if len(key) != 2:
            continue
        a, b = key
        mean_fg, var_fg = stats[(a, b)]
        a_ok = cfg.clusterable.get(a, True)
        b_ok = cfg.clusterable.get(b, True)
        n_a, n_b = schema.raw_sizes[a], schema.raw_sizes[b]
        if a_ok and b_ok:
            maps[(a, b)] = cluster_pair(mean_fg, var_fg, cfg)
        elif a_ok and not b_ok:
            cmap_a = maps[(a,)]
            maps[(a, b)] = np.repeat(cmap_a, n_b) * n_b + np.tile(np.arange(n_b, dtype=np.int32), n_a)
        elif not a_ok and b_ok:
            cmap_b = maps[(b,)]
            k_b = int(cmap_b.max()) + 1
            maps[(a, b)] = np.repeat(np.arange(n_a, dtype=np.int32), n_b) * k_b + np.tile(cmap_b, n_a)
        else:
            maps[(a, b)] = np.arange(n_a * n_b, dtype=np.int32)
    return maps


# ----------------------------
# Offsets and global slots
# ----------------------------

def compute_offsets(schema: ModelSchema, maps: Mapping[Key, ArrayI32]) -> Dict[Key, int]:
    offsets: Dict[Key, int] = {}
    cursor = 0
    for key in schema.ordered_keys():
        k = int(maps[key].max()) + 1
        offsets[key] = cursor
        cursor += k
    return offsets


# ----------------------------
# Packing & remapping
# ----------------------------

def pack_raw_params_to_clusters(
    schema: ModelSchema, maps: Mapping[Key, ArrayI32], raw_params: Mapping[Key, ArrayF64]
) -> ArrayF64:
    offsets = compute_offsets(schema, maps)
    total = sum((int(maps[key].max()) + 1) for key in schema.ordered_keys())
    theta = np.zeros(total, dtype=np.float64)
    for key in schema.ordered_keys():
        labels = maps[key]
        start = offsets[key]
        k = int(labels.max()) + 1
        sums = np.zeros(k, dtype=np.float64)
        cnts = np.zeros(k, dtype=np.int64)
        vals = raw_params[key]
        for raw_id, c in enumerate(labels):
            sums[c] += vals[raw_id]
            cnts[c] += 1
        theta[start:start + k] = sums / np.maximum(cnts, 1)
    return theta


def remap_raw_grads_to_clusters(
    schema: ModelSchema, maps: Mapping[Key, ArrayI32], raw_grads: Mapping[Key, ArrayF64]
) -> ArrayF64:
    offsets = compute_offsets(schema, maps)
    total = sum((int(maps[key].max()) + 1) for key in schema.ordered_keys())
    gflat = np.zeros(total, dtype=np.float64)
    for key in schema.ordered_keys():
        labels = maps[key]
        start = offsets[key]
        for raw_id, c in enumerate(labels):
            gflat[start + c] += raw_grads[key][raw_id]
    return gflat


# ----------------------------
# Repacking on refresh
# ----------------------------

def repack_state_on_refresh(
    schema: ModelSchema,
    old_maps: Mapping[Key, ArrayI32],
    new_maps: Mapping[Key, ArrayI32],
    old_theta: ArrayF64,
    old_m: Optional[ArrayF64] = None,
    old_v: Optional[ArrayF64] = None,
) -> Tuple[ArrayF64, Optional[ArrayF64], Optional[ArrayF64]]:
    old_offsets = compute_offsets(schema, old_maps)
    new_offsets = compute_offsets(schema, new_maps)
    new_total = sum((int(new_maps[key].max()) + 1) for key in schema.ordered_keys())
    new_theta = np.zeros(new_total, dtype=np.float64)
    new_m_arr = np.zeros(new_total, dtype=np.float64) if old_m is not None else None
    new_v_arr = np.zeros(new_total, dtype=np.float64) if old_v is not None else None
    new_cnts = np.zeros(new_total, dtype=np.int64)

    for key in schema.ordered_keys():
        n_raw = schema.raw_size_for(key)
        old_start = old_offsets[key]
        new_start = new_offsets[key]
        old_labels = old_maps[key]
        new_labels = new_maps[key]
        k_new = int(new_labels.max()) + 1
        for raw_id in range(n_raw):
            c_old = old_labels[raw_id]
            c_new = new_labels[raw_id]
            old_slot = old_start + c_old
            new_slot = new_start + c_new
            new_theta[new_slot] += old_theta[old_slot]
            if new_m_arr is not None and old_m is not None and old_v is not None:
                new_m_arr[new_slot] += old_m[old_slot]
                new_v_arr[new_slot] += old_v[old_slot]
            new_cnts[new_slot] += 1
        seg = slice(new_start, new_start + k_new)
        cnts = np.maximum(new_cnts[seg], 1)
        new_theta[seg] = new_theta[seg] / cnts
        if new_m_arr is not None:
            new_m_arr[seg] = new_m_arr[seg] / cnts
            new_v_arr[seg] = new_v_arr[seg] / cnts

    return new_theta, new_m_arr, new_v_arr


# ----------------------------
# Clustered Optimizer wrapper
# ----------------------------

class ClusteredOptimizer:
    def __init__(
        self,
        base_optimizer: Optimizer,
        schema: ModelSchema,
        cfg: ClusteringConfig,
        *,
        initial_maps: Optional[ClusterMaps] = None,
        refresh_every_steps: int = 1,
        refresh_fn: Callable[[ModelSchema, ValueStats, ClusteringConfig], ClusterMaps] = refresh_maps,
    ) -> None:
        self.opt = base_optimizer
        self.schema = schema
        self.cfg = cfg
        self.refresh_every_steps = max(1, int(refresh_every_steps))
        self.refresh_fn = refresh_fn
        self.maps = identity_maps(schema) if not cfg.cluster_enabled else (
            initial_maps if initial_maps is not None else identity_maps(schema)
        )
        self.offsets = compute_offsets(self.schema, self.maps)
        total = sum((int(self.maps[key].max()) + 1) for key in self.schema.ordered_keys())
        self.theta = np.zeros(total, dtype=np.float64)
        self.gflat = np.zeros_like(self.theta)
        if hasattr(self.opt, "m") and hasattr(self.opt, "v"):
            self.opt.m = np.zeros_like(self.theta)
            self.opt.v = np.zeros_like(self.theta)
        self._step = 0

    def scatter_from_raw_params(self, raw_params: Mapping[Key, ArrayF64]) -> None:
        self.theta = pack_raw_params_to_clusters(self.schema, self.maps, raw_params)
        if hasattr(self.opt, "m") and hasattr(self.opt, "v"):
            self.opt.m = np.zeros_like(self.theta)
            self.opt.v = np.zeros_like(self.theta)

    def step(self, raw_grads: Mapping[Key, ArrayF64], value_stats: ValueStats) -> None:
        self.gflat[:] = remap_raw_grads_to_clusters(self.schema, self.maps, raw_grads)
        self.opt.step(self.theta, self.gflat)
        self._step += 1
        if self.cfg.cluster_enabled and (self._step % self.refresh_every_steps == 0):
            new_maps = self.refresh_fn(self.schema, value_stats, self.cfg)
            if not self._maps_equal(self.maps, new_maps):
                old_m = getattr(self.opt, "m", None)
                old_v = getattr(self.opt, "v", None)
                new_theta, new_m, new_v = repack_state_on_refresh(
                    self.schema, self.maps, new_maps, self.theta, old_m, old_v
                )
                self.maps = new_maps
                self.offsets = compute_offsets(self.schema, self.maps)
                self.theta = new_theta
                self.gflat = np.zeros_like(self.theta)
                if new_m is not None and new_v is not None:
                    self.opt.m = new_m
                    self.opt.v = new_v

    @staticmethod
    def _maps_equal(a: Mapping[Key, ArrayI32], b: Mapping[Key, ArrayI32]) -> bool:
        if a.keys() != b.keys():
            return False
        return all(np.array_equal(a[k], b[k]) for k in a.keys())


# ----------------------------
# Demo
# ----------------------------

if __name__ == "__main__":
    # Minimal optimizers
    class SGD:
        def __init__(self, lr: float = 0.1) -> None:
            self.lr = float(lr)
        def step(self, params: ArrayF64, grads: ArrayF64) -> None:
            params -= self.lr * grads

    class Adam:
        def __init__(self, lr: float = 0.05, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
            self.lr, self.beta1, self.beta2, self.eps = float(lr), float(beta1), float(beta2), float(eps)
            self.t = 0
        def step(self, params: ArrayF64, grads: ArrayF64) -> None:
            if not hasattr(self, "m") or not hasattr(self, "v"):
                self.m = np.zeros_like(params)
                self.v = np.zeros_like(params)
            self.t += 1
            self.m = self.beta1 * self.m + (1.0 - self.beta1) * grads
            self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grads**2)
            mhat = self.m / (1.0 - self.beta1**self.t)
            vhat = self.v / (1.0 - self.beta2**self.t)
            params -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

    # Schema 3+2+4; pairs on (Client,Algo) and (Client,Venue)
    schema = ModelSchema(
        raw_sizes={"Client": 3, "Algo": 2, "Venue": 4},
        pairs=[("Client", "Algo"), ("Client", "Venue")],
    )

    cfg = ClusteringConfig(
        min_gap_single=0.8,
        split_thresh_single=0.6,
        min_gap_pair=1.0,
        split_thresh_pair=0.0,
        clusterable={"Client": True, "Algo": False, "Venue": True},
        null_mu=0.05,
        null_var=1e-4,
        cluster_enabled=True,
    )

    # Fake stats per key (as reducer output)
    rng = np.random.default_rng(7)
    means_client = np.array([0.12, 0.11, -0.08])
    vars_client = np.array([0.02, 0.02, 0.02])

    means_algo = np.array([0.0, 0.25])
    vars_algo = np.array([0.01, 0.02])

    means_venue = np.array([-0.35, -0.38, 0.07, 0.31])
    vars_venue = np.array([0.02, 0.02, 0.01, 0.02])

    means_client_algo = rng.normal(0.0, 0.15, schema.raw_size_for(("Client", "Algo")))
    vars_client_algo = np.full_like(means_client_algo, 0.03)

    means_client_venue = rng.normal(0.0, 0.15, schema.raw_size_for(("Client", "Venue")))
    vars_client_venue = np.full_like(means_client_venue, 0.03)

    stats: ValueStats = {
        ("Client",): (means_client, vars_client),
        ("Algo",): (means_algo, vars_algo),
        ("Venue",): (means_venue, vars_venue),
        ("Client", "Algo"): (means_client_algo, vars_client_algo),
        ("Client", "Venue"): (means_client_venue, vars_client_venue),
    }

    maps0 = refresh_maps(schema, stats, cfg)
    offsets0 = compute_offsets(schema, maps0)

    print("Initial cluster maps (local ids) and K per key:")
    for key in schema.ordered_keys():
        k = int(maps0[key].max()) + 1
        print(f"  {key}: labels={maps0[key]}  K={k}  offset={offsets0[key]}")

    # Wrap optimizer and run a couple of steps
    sgd = SGD(lr=0.1)
    opt = ClusteredOptimizer(sgd, schema, cfg, initial_maps=maps0, refresh_every_steps=2)
    raw_params0 = {k: np.zeros(schema.raw_size_for(k), dtype=float) for k in schema.ordered_keys()}
    opt.scatter_from_raw_params(raw_params0)

    raw_grads = {
        ("Client",): np.array([+1.0, -0.5, 0.0]),
        ("Algo",): np.array([+0.1, 0.0]),
        ("Venue",): np.array([0.0, 0.0, -0.2, +0.3]),
        ("Client", "Algo"): rng.normal(0, 0.2, schema.raw_size_for(("Client", "Algo"))),
        ("Client", "Venue"): rng.normal(0, 0.2, schema.raw_size_for(("Client", "Venue"))),
    }

    opt.step(raw_grads, stats)
    print(f"\nAfter step 1: theta.shape={opt.theta.shape}")

    opt.step(raw_grads, stats)
    print(f"After step 2: theta.shape={opt.theta.shape}")

    # Score an example event
    x_client, x_algo, x_venue = 2, 1, 3
    n_venue = schema.raw_sizes["Venue"]

    c_cli = opt.maps[("Client",)][x_client]
    c_alg = opt.maps[("Algo",)][x_algo]
    c_ven = opt.maps[("Venue",)][x_venue]
    s_cli = opt.offsets[("Client",)] + c_cli
    s_alg = opt.offsets[("Algo",)] + c_alg
    s_ven = opt.offsets[("Venue",)] + c_ven

    raw_pair_cv = x_client * n_venue + x_venue
    c_cv = opt.maps[("Client", "Venue")][raw_pair_cv]
    s_cv = opt.offsets[("Client", "Venue")] + c_cv

    y_hat = opt.theta[s_cli] + opt.theta[s_alg] + opt.theta[s_ven] + opt.theta[s_cv]

    print("\nExample scoring:")
    print(f"  raw ids: Client={x_client}, Algo={x_algo}, Venue={x_venue}")
    print(f"  local ids: Client={c_cli}, Algo={c_alg}, Venue={c_ven}, Client×Venue={c_cv}")
    print(f"  global slots: {s_cli}, {s_alg}, {s_ven}, {s_cv}")
    print(f"  y_hat: {y_hat:.6f}")

    # Sanity checks
    for key in schema.ordered_keys():
        assert maps0[key].shape[0] == schema.raw_size_for(key)
    last = 0
    for key in schema.ordered_keys():
        assert opt.offsets[key] == last
        last += int(opt.maps[key].max()) + 1

    print("\n✅ Demo finished successfully.")
