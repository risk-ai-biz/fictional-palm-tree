"""
Upward-propagating order fills with Polars preprocessing and Numba kernels.

Layout: compact per-(day, symbol) datasets packaged into contiguous "arena"
arrays, with an offsets table so each dataset slice is contiguous.

You can:
  1) Build the arena from Polars (Parquet or in-memory).
  2) Run many simulation passes (e.g., gradient steps) with fast O(N) resets.

---------------------------------------------------------------------------
Symbols
  D  := number of trading days
  S  := number of symbols per day
  M  := number of (day, symbol) datasets (oppsets), M = count of unique pairs
  Nm := number of orders in dataset m  (varies per dataset)
  N  := sum_m Nm (total orders across all datasets)

Arrays (global/arena, aligned by row):
  order_id_flat        : (N,)   int32    # original order id (unique per day)
  parent_idx_flat      : (N,)   int32    # LOCAL parent row index in dataset, or -1
  baseline_qty_flat    : (N,)   float64  # initial quantities per order
  workspace_qty_flat   : (N,)   float64  # scratch quantities (reset each run)

Dataset index (arena slicing):
  ds_offsets           : (M+1,) int64    # dataset m rows are [ds_offsets[m] : ds_offsets[m+1])
  ds_day               : (M,)   int32    # day id per dataset (for reference / lookup)
  ds_sym_id            : (M,)   int32    # symbol id per dataset (int-mapped)

All simulation kernels operate on slices derived from ds_offsets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
from numba import njit
# Polars is used only for preprocessing (not in kernels)
import polars as pl

ROOT = np.int32(-1)


# ============================================================================
# Arena container (plain NumPy — Numba can consume its fields)
# ============================================================================

@dataclass
class Arena:
    """
    Packed arrays for all (day, symbol) datasets.

    Shapes / dtypes:
      order_id_flat      : (N,)   int32
      parent_idx_flat    : (N,)   int32
      baseline_qty_flat  : (N,)   float64
      workspace_qty_flat : (N,)   float64

      ds_offsets         : (M+1,) int64
      ds_day             : (M,)   int32
      ds_sym_id          : (M,)   int32

    N = total number of rows across all datasets.
    M = number of datasets (unique day, symbol pairs).
    """
    order_id_flat: np.ndarray
    parent_idx_flat: np.ndarray
    baseline_qty_flat: np.ndarray
    workspace_qty_flat: np.ndarray
    ds_offsets: np.ndarray
    ds_day: np.ndarray
    ds_sym_id: np.ndarray

    def num_rows(self) -> int:
        return int(self.order_id_flat.shape[0])

    def num_datasets(self) -> int:
        return int(self.ds_offsets.shape[0] - 1)


# ============================================================================
# Polars → Arena
# ============================================================================

def _ensure_symbol_ids(df: pl.DataFrame, sym_col: str = "symbol") -> pl.DataFrame:
    """
    Map string symbols to dense int codes (sym_id: int32).
    If `symbol` already int-like, just rename to `sym_id`.
    """
    if pl.datatypes.is_utf8(df.schema[sym_col]):
        return df.with_columns(
            pl.col(sym_col).cast(pl.Categorical).to_physical().cast(pl.Int32).alias("sym_id")
        )
    else:
        return df.rename({sym_col: "sym_id"}).with_columns(pl.col("sym_id").cast(pl.Int32))


def build_arena_from_polars(
    df_raw: pl.DataFrame,
    *,
    day_col: str = "day",
    symbol_col: str = "symbol",
    order_col: str = "order_id",
    parent_col: str = "parent_order_id",
    qty_col: str = "baseline_qty",
) -> Arena:
    """
    Preprocess with Polars to produce a compact arena of arrays.
    Parent mapping is *local to each (day, symbol)* dataset.

    Required columns (any dtype that can cast to target types):
      day_col    -> int32
      symbol_col -> str or int32 (mapped to sym_id:int32)
      order_col  -> int32  (unique per day)
      parent_col -> int32  (-1 for roots or order_id of the parent)
      qty_col    -> float64

    Returns
    -------
    Arena
    """
    # Canonicalize & narrow types
    df = df_raw.with_columns(
        pl.col(day_col).cast(pl.Int32).alias("day"),
        pl.col(order_col).cast(pl.Int32).alias("order_id"),
        pl.col(parent_col).cast(pl.Int32).alias("parent_order_id"),
        pl.col(qty_col).cast(pl.Float64).alias("baseline_qty"),
    )
    df = _ensure_symbol_ids(df.rename({symbol_col: "symbol"}))
    df = df.select("day", "sym_id", "order_id", "parent_order_id", "baseline_qty")

    # Per-(day, sym) local row index: 0..Nm-1
    df = df.sort(["day", "sym_id", "order_id"])
    df = df.with_columns(pl.cum_count().over(["day", "sym_id"]).cast(pl.Int32).alias("local_row"))

    # Self-join to resolve parent local_row within each (day, sym)
    parent_map = df.select(
        pl.col("day"),
        pl.col("sym_id"),
        pl.col("order_id").alias("parent_order_id"),
        pl.col("local_row").alias("parent_local_row"),
    )

    df = df.join(
        parent_map,
        on=["day", "sym_id", "parent_order_id"],
        how="left",
    ).with_columns(
        pl.col("parent_local_row").fill_null(ROOT).cast(pl.Int32)
    )

    # Sort in final arena order: (day, sym, local_row)
    df = df.sort(["day", "sym_id", "local_row"])

    # Dataset counts (in the same (day, sym) order)
    groups = (
        df.group_by(["day", "sym_id"])
          .len()
          .sort(["day", "sym_id"])
    )

    counts = groups["len"].to_numpy().astype(np.int64)           # (M,)
    ds_day = groups["day"].to_numpy().astype(np.int32)           # (M,)
    ds_sym_id = groups["sym_id"].to_numpy().astype(np.int32)     # (M,)

    ds_offsets = np.zeros(counts.shape[0] + 1, dtype=np.int64)   # (M+1,)
    np.cumsum(counts, out=ds_offsets[1:])

    # Flattened columns (aligned to arena rows)
    order_id_flat = df["order_id"].to_numpy().astype(np.int32)             # (N,)
    parent_idx_flat = df["parent_local_row"].to_numpy().astype(np.int32)   # (N,)
    baseline_qty_flat = df["baseline_qty"].to_numpy().astype(np.float64)   # (N,)
    workspace_qty_flat = np.empty_like(baseline_qty_flat)                  # (N,)

    return Arena(
        order_id_flat=order_id_flat,
        parent_idx_flat=parent_idx_flat,
        baseline_qty_flat=baseline_qty_flat,
        workspace_qty_flat=workspace_qty_flat,
        ds_offsets=ds_offsets,
        ds_day=ds_day,
        ds_sym_id=ds_sym_id,
    )


# ============================================================================
# Numba kernels (hot path)
# ============================================================================

@njit
def propagate_up(order_idx: np.int32,
                 fill_qty: float,
                 remaining_qty: np.ndarray,   # (Nm,) float64  (slice)
                 parent_idx:   np.ndarray):   # (Nm,) int32    (slice)
    """
    Subtract `fill_qty` from `order_idx` and all its ancestors (upward only).
    """
    remaining_qty[order_idx] -= fill_qty
    p = parent_idx[order_idx]
    while p != ROOT:
        remaining_qty[p] -= fill_qty
        p = parent_idx[p]


@njit
def reset_dataset_state(workspace_slice: np.ndarray,  # (Nm,) float64
                        baseline_slice:  np.ndarray):  # (Nm,) float64
    """
    Reset per-dataset workspace to baseline (memcpy-like).
    """
    np.copyto(workspace_slice, baseline_slice)


@njit
def apply_fills_in_dataset(fill_indices: np.ndarray,  # (K,) int32  local row indices
                           fill_qtys:    np.ndarray,  # (K,) float64
                           remaining_qty: np.ndarray, # (Nm,) float64
                           parent_idx:    np.ndarray  # (Nm,) int32
                           ):
    """
    Apply K fills sequentially in one dataset slice and propagate upward.
    """
    K = fill_indices.shape[0]
    for k in range(K):
        i = fill_indices[k]
        q = fill_qtys[k]
        propagate_up(i, q, remaining_qty, parent_idx)


@njit
def run_one_dataset(ds_id: int,
                    ds_offsets: np.ndarray,        # (M+1,) int64
                    parent_idx_flat: np.ndarray,   # (N,) int32
                    baseline_qty_flat: np.ndarray, # (N,) float64
                    workspace_qty_flat: np.ndarray,# (N,) float64
                    fill_indices: np.ndarray,      # (K,) int32 (local)
                    fill_qtys: np.ndarray          # (K,) float64
                    ):
    """
    Run a single dataset identified by ds_id.

    Slices:
      start = ds_offsets[ds_id]; end = ds_offsets[ds_id+1]
      parent_idx_local   := parent_idx_flat[start:end]     (Nm,)
      baseline_local     := baseline_qty_flat[start:end]   (Nm,)
      workspace_local    := workspace_qty_flat[start:end]  (Nm,)
    """
    start = ds_offsets[ds_id]
    end = ds_offsets[ds_id + 1]

    parent_idx_local = parent_idx_flat[start:end]
    baseline_local   = baseline_qty_flat[start:end]
    workspace_local  = workspace_qty_flat[start:end]

    reset_dataset_state(workspace_local, baseline_local)
    apply_fills_in_dataset(fill_indices, fill_qtys, workspace_local, parent_idx_local)


@njit
def run_all_datasets(ds_offsets: np.ndarray,        # (M+1,) int64
                     parent_idx_flat: np.ndarray,   # (N,) int32
                     baseline_qty_flat: np.ndarray, # (N,) float64
                     workspace_qty_flat: np.ndarray,# (N,) float64
                     # Per-dataset fills packed as a typed list (length M).
                     fill_idx_list,                 # List[np.ndarray[int32]]
                     fill_qty_list                  # List[np.ndarray[float64]]
                     ):
    """
    Run every dataset once. `fill_idx_list[m]` and `fill_qty_list[m]`
    are the fills for dataset m (local row indices).
    """
    M = ds_offsets.shape[0] - 1
    for m in range(M):
        run_one_dataset(
            m, ds_offsets, parent_idx_flat, baseline_qty_flat, workspace_qty_flat,
            fill_idx_list[m], fill_qty_list[m]
        )


# ============================================================================
# Validation helpers (optional)
# ============================================================================

@njit
def validate_forest(parent_idx: np.ndarray) -> np.bool_:
    """
    Validate a parent array encodes a forest (no cycles).
    Works on a single dataset slice (use on parent_idx_flat[start:end]).
    """
    N = parent_idx.shape[0]
    state = np.zeros(N, dtype=np.uint8)  # 0=unseen, 1=visiting, 2=done
    for i in range(N):
        node = i
        while node != ROOT and state[node] == 0:
            state[node] = 1
            node = parent_idx[node]
        if node != ROOT and state[node] == 1:
            return False
        # mark path as done
        node2 = i
        while node2 != ROOT and state[node2] == 1:
            state[node2] = 2
            node2 = parent_idx[node2]
    return True


# ============================================================================
# Example data / tests
# ============================================================================

def _example_polars_df() -> pl.DataFrame:
    """
    Build a tiny, readable example:
      Days:   2 (d=0,1)
      Symbols: map 'AAA'->0, 'BBB'->1 automatically via Categorical→int
      Each (day, sym) is a small tree.

    Columns:
      day:int, symbol:str, order_id:int, parent_order_id:int(-1 for root), baseline_qty:float
    """
    data = {
        "day":   [0,0,0,0,  0,0,  1,1,1],
        "symbol":["AAA","AAA","AAA","AAA", "BBB","BBB", "AAA","AAA","BBB"],
        "order_id":[10,11,12,13,  20,21,  30,31,  40],
        "parent_order_id":[-1,10,10,11,  -1,20,  -1,30,  -1],
        "baseline_qty":[2.0,1.5,0.7,1.0,  3.0,0.5,  1.2,0.9,  4.0],
    }
    return pl.DataFrame(data)


def _example_build_and_run():
    # Build arena
    df = _example_polars_df()
    arena = build_arena_from_polars(df)

    print("M datasets:", arena.num_datasets(), "N total rows:", arena.num_rows())
    print("ds_day   :", arena.ds_day.tolist())
    print("ds_sym_id:", arena.ds_sym_id.tolist())
    print("ds_offsets:", arena.ds_offsets.tolist())

    # Create per-dataset fills (local indices)
    # For demo, fill index 0 and 1 by 0.5 for each dataset that has >=2 rows
    from numba.typed import List
    M = arena.num_datasets()

    fill_idx_list = List()
    fill_qty_list = List()
    for m in range(M):
        start = arena.ds_offsets[m]
        end = arena.ds_offsets[m+1]
        Nm = end - start

        if Nm >= 2:
            fill_idx = np.array([1, 0], dtype=np.int32)  # local rows
            fill_qty = np.array([0.5, 0.25], dtype=np.float64)
        elif Nm == 1:
            fill_idx = np.array([0], dtype=np.int32)
            fill_qty = np.array([0.1], dtype=np.float64)
        else:
            fill_idx = np.empty(0, dtype=np.int32)
            fill_qty = np.empty(0, dtype=np.float64)

        fill_idx_list.append(fill_idx)
        fill_qty_list.append(fill_qty)

    # Warm up JIT
    run_all_datasets(
        arena.ds_offsets,
        arena.parent_idx_flat,
        arena.baseline_qty_flat,
        arena.workspace_qty_flat,
        fill_idx_list,
        fill_qty_list,
    )

    # Run once more (timing would go here IRL)
    run_all_datasets(
        arena.ds_offsets,
        arena.parent_idx_flat,
        arena.baseline_qty_flat,
        arena.workspace_qty_flat,
        fill_idx_list,
        fill_qty_list,
    )

    # Inspect first dataset slice
    m = 0
    s, e = arena.ds_offsets[m], arena.ds_offsets[m+1]
    print("\nDataset 0 rows:", int(e-s))
    print("order_id         :", arena.order_id_flat[s:e].tolist())
    print("parent_idx(local):", arena.parent_idx_flat[s:e].tolist())
    print("baseline         :", np.round(arena.baseline_qty_flat[s:e], 3).tolist())
    print("after run        :", np.round(arena.workspace_qty_flat[s:e], 3).tolist())

    # Validate forest on the first dataset
    ok = validate_forest(arena.parent_idx_flat[s:e])
    print("parent forest valid?", bool(ok))


if __name__ == "__main__":
    _example_build_and_run()
