
"""
training_tracker.py

An extended, deterministic tracking harness for clustered_optimizer that you can
drop into your training loop. It:

- Runs a reproducible multi-epoch training demo (you can replace the loop with yours).
- Logs per-epoch:
  - Cluster counts per key
  - ARI (Adjusted Rand Index) stability vs previous epoch per key
  - Theta L2 step size
  - Per-key WCSS (within-cluster sum of squares) based on epoch stats
- Saves:
  - CSVs (and Parquet if pyarrow is available)
  - Transition matrices (prev→curr) for each key as CSV
  - Heatmaps for each transition matrix
  - Per-epoch cluster maps to .npz for perfect reproducibility
  - A simple INDEX.md linking to artifacts

Charts: matplotlib only, one chart per figure, default styles.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import clustered_optimizer as co

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "tracker_ext")
os.makedirs(OUT, exist_ok=True)

# ---------------------------
# Optional Parquet writer
# ---------------------------
def to_parquet_if_possible(df: pd.DataFrame, path: str) -> None:
    try:
        import pyarrow  # noqa: F401
        df.to_parquet(path, index=False)
    except Exception:
        # Fall back silently; CSV will still be saved by caller.
        pass

# ---------------------------
# Metrics
# ---------------------------
def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    lt = labels_true.astype(int)
    lp = labels_pred.astype(int)
    def relabel(a: np.ndarray) -> np.ndarray:
        uniq = np.unique(a)
        mapping = {old: i for i, old in enumerate(uniq)}
        return np.array([mapping[x] for x in a], dtype=int)
    lt = relabel(lt)
    lp = relabel(lp)

    n = lt.size
    kt = int(lt.max()) + 1
    kp = int(lp.max()) + 1
    nij = np.zeros((kt, kp), dtype=np.int64)
    for i in range(n):
        nij[lt[i], lp[i]] += 1
    ai = nij.sum(axis=1)
    bj = nij.sum(axis=0)

    def comb2(x: np.ndarray) -> float:
        return float((x * (x - 1)).sum() / 2.0)
    sum_comb_nij = comb2(nij.astype(float))
    sum_comb_ai = comb2(ai.astype(float))
    sum_comb_bj = comb2(bj.astype(float))
    total_comb = comb2(np.array([n], dtype=float))
    expected_index = (sum_comb_ai * sum_comb_bj) / total_comb if total_comb != 0 else 0.0
    max_index = 0.5 * (sum_comb_ai + sum_comb_bj)
    index = sum_comb_nij
    denom = (max_index - expected_index)
    if denom == 0:
        return 1.0
    return (index - expected_index) / denom

def transition_counts(prev_labels: np.ndarray, curr_labels: np.ndarray) -> np.ndarray:
    """Return count matrix T where T[i,j] is #raws moved from prev cluster i to curr cluster j."""
    prev = prev_labels.astype(int)
    curr = curr_labels.astype(int)
    k_prev = int(prev.max()) + 1
    k_curr = int(curr.max()) + 1
    mat = np.zeros((k_prev, k_curr), dtype=np.int64)
    for idx in range(prev.shape[0]):
        mat[prev[idx], curr[idx]] += 1
    return mat

def wcss(values: np.ndarray, labels: np.ndarray) -> float:
    """Within-cluster sum-of-squares for a single key given raw 'values' and local labels."""
    lab = labels.astype(int)
    k = int(lab.max()) + 1
    ss = 0.0
    for c in range(k):
        idx = np.where(lab == c)[0]
        if idx.size == 0:
            continue
        mu = float(np.mean(values[idx]))
        ss += float(((values[idx] - mu) ** 2).sum())
    return ss

# ---------------------------
# Deterministic RNG helpers
# ---------------------------
def save_rng_state(rng: np.random.Generator) -> str:
    return json.dumps(rng.bit_generator.state)

def load_rng_state(rng: np.random.Generator, state_json: str) -> None:
    rng.bit_generator.state = json.loads(state_json)

# ---------------------------
# Demo schema & config
# ---------------------------
schema = co.ModelSchema(
    raw_sizes={"Client": 3, "Algo": 2, "Venue": 4},
    pairs=[("Client", "Venue")]
)
cfg = co.ClusteringConfig(
    min_gap_single=0.9,
    split_thresh_single=0.5,
    min_gap_pair=1.1,
    split_thresh_pair=0.0,
    clusterable={"Client": True, "Algo": True, "Venue": True},
    null_mu=0.03,
    null_var=1e-6,
    cluster_enabled=True,
)

def make_stats_for_epoch(rng: np.random.Generator, epoch: int) -> co.ValueStats:
    # Singles
    mc = np.array([0.10, 0.08, -0.06]) + 0.01 * math.sin(0.3 * epoch)
    vc = np.array([0.02, 0.02, 0.02])

    ma = np.array([0.02, 0.18]) + 0.005 * math.cos(0.2 * epoch)
    va = np.array([0.01, 0.02])

    mv = np.array([-0.32, -0.34, 0.06, 0.26]) + 0.01 * math.sin(0.5 * epoch)
    vv = np.array([0.02, 0.02, 0.01, 0.02])

    # Pair
    n_cv = schema.raw_size_for(("Client", "Venue"))
    base = np.linspace(-0.2, 0.2, n_cv)
    ripple = 0.03 * np.sin(np.linspace(0, 2*np.pi, n_cv, endpoint=False) + 0.25*epoch)
    mcv = base + ripple
    vcv = np.full(n_cv, 0.03)

    return {
        ("Client",): (mc, vc),
        ("Algo",): (ma, va),
        ("Venue",): (mv, vv),
        ("Client", "Venue"): (mcv, vcv),
    }

# ---------------------------
# Training loop (demo)
# ---------------------------
class SGD:
    def __init__(self, lr: float = 0.05) -> None:
        self.lr = float(lr)
    def step(self, params: np.ndarray, grads: np.ndarray) -> None:
        params -= self.lr * grads

def main() -> None:
    EPOCHS = 10
    rng = np.random.default_rng(999)
    rng_snapshot = save_rng_state(rng)

    stats0 = make_stats_for_epoch(rng, 0)
    maps0 = co.refresh_maps(schema, stats0, cfg)
    opt = co.ClusteredOptimizer(SGD(lr=0.05), schema, cfg, initial_maps=maps0, refresh_every_steps=1)
    opt.scatter_from_raw_params({k: np.zeros(schema.raw_size_for(k), float) for k in schema.ordered_keys()})

    # Logs
    counts_rows: List[Dict[str, float]] = []
    ari_rows: List[Dict[str, float]] = []
    theta_rows: List[Dict[str, float]] = []
    wcss_rows: List[Dict[str, float]] = []

    maps_prev = maps0

    for epoch in range(EPOCHS):
        stats = make_stats_for_epoch(rng, epoch)

        # Synthetic, reproducible grads per key
        load_rng_state(rng, rng_snapshot)
        raw_grads = {}
        for key in schema.ordered_keys():
            n_raw = schema.raw_size_for(key)
            g = np.sin(0.1 * epoch + np.arange(n_raw) * 0.05) * 0.2
            raw_grads[key] = g.astype(float)

        theta_prev = opt.theta.copy()
        opt.step(raw_grads, stats)

        # ---- Metrics
        # counts
        counts = {str(key): int(opt.maps[key].max()) + 1 for key in schema.ordered_keys()}
        counts_rows.append({"epoch": epoch, **counts})

        # ARI
        if epoch > 0:
            aris = {str(key): adjusted_rand_index(maps_prev[key], opt.maps[key]) for key in schema.ordered_keys()}
            ari_rows.append({"epoch": epoch, **aris})

        # WCSS from this epoch's stats and current labels
        wcss_vals = {}
        for key in schema.ordered_keys():
            mean_vals = stats[key][0]
            labels = opt.maps[key]
            wcss_vals[str(key)] = wcss(mean_vals, labels)
        wcss_rows.append({"epoch": epoch, **wcss_vals})

        # Theta L2
        theta_rows.append({"epoch": epoch, "theta_l2": float(np.linalg.norm(opt.theta - theta_prev))})

        # ---- Transition matrix & heatmap per key
        if epoch > 0:
            for key in schema.ordered_keys():
                mat = transition_counts(maps_prev[key], opt.maps[key])
                # CSV
                csv_path = os.path.join(OUT, f"transition_{str(key).replace(' ', '')}_e{epoch-1}_to_e{epoch}.csv")
                pd.DataFrame(mat).to_csv(csv_path, index=False)
                # Heatmap
                fig = plt.figure()
                plt.imshow(mat, aspect="auto")
                plt.title(f"Transition counts {key}: e{epoch-1} → e{epoch}")
                plt.xlabel("current cluster")
                plt.ylabel("previous cluster")
                plt.tight_layout()
                fig.savefig(os.path.join(OUT, f"transition_{str(key).replace(' ', '')}_e{epoch-1}_to_e{epoch}.png"))
                plt.close(fig)

        # Save per-epoch maps snapshot for perfect reproducibility
        npz_path = os.path.join(OUT, f"maps_epoch_{epoch}.npz")
        np.savez_compressed(npz_path, **{f"k_{i}": opt.maps[k] for i, k in enumerate(schema.ordered_keys())})

        maps_prev = {k: opt.maps[k].copy() for k in schema.ordered_keys()}

    # ---- Write log tables
    df_counts = pd.DataFrame(counts_rows)
    df_counts.to_csv(os.path.join(OUT, "cluster_counts.csv"), index=False)
    to_parquet_if_possible(df_counts, os.path.join(OUT, "cluster_counts.parquet"))

    if ari_rows:
        df_ari = pd.DataFrame(ari_rows)
        df_ari.to_csv(os.path.join(OUT, "ari_prev_epoch.csv"), index=False)
        to_parquet_if_possible(df_ari, os.path.join(OUT, "ari_prev_epoch.parquet"))

    df_theta = pd.DataFrame(theta_rows)
    df_theta.to_csv(os.path.join(OUT, "theta_l2.csv"), index=False)
    to_parquet_if_possible(df_theta, os.path.join(OUT, "theta_l2.parquet"))

    df_wcss = pd.DataFrame(wcss_rows)
    df_wcss.to_csv(os.path.join(OUT, "wcss.csv"), index=False)
    to_parquet_if_possible(df_wcss, os.path.join(OUT, "wcss.parquet"))

    # ---- Summary charts
    # Cluster counts
    for key in schema.ordered_keys():
        fig = plt.figure()
        xs = df_counts["epoch"].values
        ys = df_counts[str(key)].values
        plt.plot(xs, ys, marker="o")
        plt.title(f"K over epochs: {key}")
        plt.xlabel("Epoch")
        plt.ylabel("K (clusters)")
        plt.tight_layout()
        fig.savefig(os.path.join(OUT, f"clusters_{str(key).replace(' ', '')}.png"))
        plt.close(fig)

    # ARI
    if len(ari_rows) > 0:
        df_ari = pd.DataFrame(ari_rows)
        for key in schema.ordered_keys():
            col = str(key)
            if col in df_ari.columns:
                fig = plt.figure()
                xs = df_ari["epoch"].values
                ys = df_ari[col].values
                plt.plot(xs, ys, marker="o")
                plt.title(f"ARI vs prev epoch: {key}")
                plt.xlabel("Epoch")
                plt.ylabel("Adjusted Rand Index")
                plt.ylim(-0.1, 1.05)
                plt.tight_layout()
                fig.savefig(os.path.join(OUT, f"ari_{str(key).replace(' ', '')}.png"))
                plt.close(fig)

    # Theta L2
    fig = plt.figure()
    plt.plot(df_theta["epoch"].values, df_theta["theta_l2"].values, marker="o")
    plt.title("Theta L2 change per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("||Δθ||₂")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "theta_l2.png"))
    plt.close(fig)

    # Final pairwise heatmap
    if ("Client", "Venue") in schema.ordered_keys():
        final_pair = maps_prev[("Client", "Venue")]
        n_c, n_v = schema.raw_sizes["Client"], schema.raw_sizes["Venue"]
        fig = plt.figure()
        plt.imshow(final_pair.reshape(n_c, n_v), aspect="auto")
        plt.title("Final pairwise cluster labels: (Client × Venue)")
        plt.xlabel("Venue raw id")
        plt.ylabel("Client raw id")
        plt.tight_layout()
        fig.savefig(os.path.join(OUT, "heatmap_client_venue.png"))
        plt.close(fig)

    # Simple index for your dashboard
    with open(os.path.join(OUT, "INDEX.md"), "w") as f:
        f.write("# Clustered Optimizer Tracking Artifacts\n\n")
        f.write("## Tables\n")
        for name in ["cluster_counts.csv", "ari_prev_epoch.csv", "theta_l2.csv", "wcss.csv"]:
            if os.path.exists(os.path.join(OUT, name)):
                f.write(f"- {name}\n")
        f.write("\n## Charts\n")
        for fname in sorted(os.listdir(OUT)):
            if fname.endswith(".png"):
                f.write(f"- {fname}\n")

if __name__ == "__main__":
    main()
