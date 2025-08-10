
"""
oppsim_consolidated.py

Upward-only fill propagation for hierarchical orders with:
- Polars preprocessing (from Parquet or in-memory DataFrame)
- Global "arena" of contiguous NumPy arrays (NumPy bundle for fast reloads)
- Per-(date, sym) oppset indexing (global & batch-friendly)
- Monolithic Numba kernels (stable across Numba versions)
- Validation, batching, and fills-packing utilities
- End-to-end examples

Files produced (EXACTLY) by save_arena(...):
  1) oppset_index.parquet
  2) arena_core.npz
  3) features.npy
  4) workspace_init.npy   (optional; only if persist_workspace=True)

===============================================================================
Schema expected in your Polars DataFrames:
  date            : pl.Date
  sym             : pl.Utf8
  OrderId         : pl.Int64
  ParentId        : pl.Int64
  RootParentId    : pl.Int64
  RemainingQty    : pl.Float64   (configurable via qty_col)
  ft_*            : feature columns (categorical / string / bool / numeric)
===============================================================================

Arena arrays (global, contiguous across (date, sym) oppsets):
  order_id_flat      : (N,)   int32
  parent_idx_flat    : (N,)   int32        # local indices within oppset; ROOT = -1
  baseline_qty_flat  : (N,)   float64
  workspace_qty_flat : (N,)   float64      # allocated at load/run time
  feature_mat_flat   : (N,F)  float32

Oppset index:
  ds_offsets         : (M+1,) int64        # row span for oppset m: [s:e)
  ds_day             : (M,)   int32        # date as days since epoch (fast slicing)
  ds_sym_id          : (M,)   int32        # dense int id of symbol
  oppset_index.parquet columns: [oppset_id, date, sym, day_i32, sym_id, start, end, n_rows]

Numba kernels:
  run_all_datasets_njit(...)
  run_subset_datasets_njit(...)

Root rule:
  ROOT = -1 when (ParentId < 0) or (ParentId == OrderId) or (ParentId == RootParentId).
  If strict_parent=True, any remaining unresolved ParentId inside group → error.
  If strict_parent=False, unresolved ParentId coerced to ROOT.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional
from pathlib import Path
import json

import numpy as np

# Optional dependencies
try:
    import polars as pl
    HAVE_POLARS = True
except Exception:
    HAVE_POLARS = False

try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        def deco(fn): return fn
        if args and callable(args[0]): return args[0]
        return deco

ROOT = np.int32(-1)  # sentinel


# ============================================================================
# Data container
# ============================================================================

@dataclass
class Arena:
    """
    Global, contiguous arrays for all oppsets in a date range.

    Shapes / dtypes:
      order_id_flat      : (N,)   int32
      parent_idx_flat    : (N,)   int32
      baseline_qty_flat  : (N,)   float64
      workspace_qty_flat : (N,)   float64
      feature_mat_flat   : (N,F)  float32

      ds_offsets : (M+1,) int64
      ds_day     : (M,)   int32
      ds_sym_id  : (M,)   int32

    N = total rows across oppsets; M = #oppsets in range; F = #features (ft_*).
    """
    order_id_flat:      np.ndarray
    parent_idx_flat:    np.ndarray
    baseline_qty_flat:  np.ndarray
    workspace_qty_flat: np.ndarray
    feature_mat_flat:   np.ndarray

    ds_offsets: np.ndarray
    ds_day:     np.ndarray
    ds_sym_id:  np.ndarray

    feature_names: List[str]

    def num_rows(self) -> int:
        return int(self.order_id_flat.shape[0])

    def num_datasets(self) -> int:
        return int(self.ds_offsets.shape[0] - 1)


# ============================================================================
# Polars → Arena (schema-aware)
# ============================================================================

def _prepare_df_schema(
    df_raw: "pl.DataFrame",
    *,
    date_col: str, sym_col: str,
    order_col: str, parent_col: str, root_parent_col: str,
    qty_col: str, feature_prefix: str
) -> "pl.DataFrame":
    """
    Cast & select required columns; keep ft_* features.
    Adds:
      sym_id:int32 (Categorical physical), day_i32:int32 (days since epoch),
      local_row:int32 (per (day_i32, sym_id) group)
    """
    if not HAVE_POLARS:
        raise RuntimeError("Polars not available. Install polars to use this builder.")

    keep_cols = [
        pl.col(date_col).cast(pl.Date).alias("date"),
        pl.col(sym_col).cast(pl.Utf8).alias("sym"),
        pl.col(order_col).cast(pl.Int64).alias("OrderId"),
        pl.col(parent_col).cast(pl.Int64).alias("ParentId"),
        pl.col(root_parent_col).cast(pl.Int64).alias("RootParentId"),
        pl.col(qty_col).cast(pl.Float64).alias("RemainingQty"),
    ] + [pl.col(c) for c in df_raw.columns if c.startswith(feature_prefix)]
    df = df_raw.select(*keep_cols)

    df = df.with_columns(
        pl.col("sym").cast(pl.Categorical).to_physical().cast(pl.Int32).alias("sym_id"),
        pl.col("date").cast(pl.Int32).alias("day_i32"),
    )
    df = df.sort(["day_i32", "sym_id", "OrderId"])
    df = df.with_columns(
        pl.cum_count().over(["day_i32", "sym_id"]).cast(pl.Int32).alias("local_row")
    )
    return df


def build_arena_from_polars_df(
    df_raw: "pl.DataFrame",
    *,
    date_col: str = "date",
    sym_col: str = "sym",
    order_col: str = "OrderId",
    parent_col: str = "ParentId",
    root_parent_col: str = "RootParentId",
    qty_col: str = "RemainingQty",
    feature_prefix: str = "ft_",
    date_start: str | "pl.Date" | None = None,
    date_end:   str | "pl.Date" | None = None,
    strict_parent: bool = True,
) -> tuple[Arena, "pl.DataFrame"]:
    """
    Build global arena from an eager Polars DataFrame with your schema.
    Returns (arena, oppset_index_df).
    """
    if not HAVE_POLARS:
        raise RuntimeError("Polars not available.")

    df = df_raw
    if date_start is not None:
        df = df.filter(pl.col(date_col) >= pl.lit(date_start).cast(pl.Date))
    if date_end is not None:
        df = df.filter(pl.col(date_col) <= pl.lit(date_end).cast(pl.Date))

    df = _prepare_df_schema(
        df, date_col=date_col, sym_col=sym_col,
        order_col=order_col, parent_col=parent_col, root_parent_col=root_parent_col,
        qty_col=qty_col, feature_prefix=feature_prefix
    )

    # ParentId -> local index within oppset
    parent_map = df.select(
        pl.col("day_i32"), pl.col("sym_id"),
        pl.col("OrderId").alias("ParentId"),
        pl.col("local_row").alias("parent_local_row"),
    )
    df = df.join(parent_map, on=["day_i32", "sym_id", "ParentId"], how="left")

    # Root logic
    root_mask = (
        (pl.col("ParentId") < 0) |
        (pl.col("ParentId") == pl.col("OrderId")) |
        (pl.col("ParentId") == pl.col("RootParentId"))
    )
    if strict_parent:
        missing = df.filter(~root_mask & pl.col("parent_local_row").is_null())
        if missing.height > 0:
            raise ValueError(
                "Found ParentId not present within its (date, sym) group. "
                "Set strict_parent=False to coerce missing parents to ROOT."
            )
        df = df.with_columns(
            pl.when(root_mask).then(pl.lit(-1))
              .otherwise(pl.col("parent_local_row"))
              .cast(pl.Int32).alias("parent_idx_local")
        )
    else:
        df = df.with_columns(
            pl.when(root_mask | pl.col("parent_local_row").is_null()).then(pl.lit(-1))
              .otherwise(pl.col("parent_local_row"))
              .cast(pl.Int32).alias("parent_idx_local")
        )

    # Oppset index
    grp = df.group_by(["day_i32", "sym_id"]).len().sort(["day_i32", "sym_id"])
    counts     = grp["len"].to_numpy().astype(np.int64)     # (M,)
    ds_day     = grp["day_i32"].to_numpy().astype(np.int32) # (M,)
    ds_sym_id  = grp["sym_id"].to_numpy().astype(np.int32)  # (M,)
    ds_offsets = np.zeros(counts.shape[0] + 1, dtype=np.int64); np.cumsum(counts, out=ds_offsets[1:])

    firsts = (
        df.sort(["day_i32", "sym_id", "local_row"])
          .group_by(["day_i32", "sym_id"])
          .agg(pl.col("date").first().alias("date"),
               pl.col("sym").first().alias("sym"))
          .sort(["day_i32", "sym_id"])
          .with_columns(pl.int_range(0, pl.len()).cast(pl.Int64).alias("oppset_id"))
    )
    oppset_index = pl.DataFrame({
        "oppset_id": firsts["oppset_id"],
        "date": firsts["date"],
        "sym": firsts["sym"],
        "day_i32": firsts["day_i32"],
        "sym_id": firsts["sym_id"],
        "start": pl.Series(ds_offsets[:-1]),
        "end":   pl.Series(ds_offsets[1:]),
        "n_rows": pl.Series(counts),
    })

    # Final row order for arena
    df = df.sort(["day_i32", "sym_id", "local_row"])

    # Core arrays
    order_id_flat      = df["OrderId"].to_numpy().astype(np.int64)
    parent_idx_flat    = df["parent_idx_local"].to_numpy().astype(np.int32)
    baseline_qty_flat  = df["RemainingQty"].to_numpy().astype(np.float64)

    if order_id_flat.size and order_id_flat.max() <= np.iinfo(np.int32).max and order_id_flat.min() >= np.iinfo(np.int32).min:
        order_id_flat = order_id_flat.astype(np.int32)

    # Feature matrix
    feat_cols = [c for c in df.columns if c.startswith(feature_prefix)]
    feature_names = feat_cols
    feat_arrays: List[np.ndarray] = []
    for c in feat_cols:
        dt = df.schema[c]
        if dt == pl.Utf8 or dt == pl.Categorical:
            col = df[c].cast(pl.Categorical).to_physical().cast(pl.Int32)
        elif dt == pl.Boolean:
            col = df[c].cast(pl.Int32)
        elif dt in (pl.Int64, pl.Int32, pl.UInt32, pl.UInt64):
            col = df[c].cast(pl.Int32)
        elif dt in (pl.Float32, pl.Float64):
            col = df[c].cast(pl.Float32)
        else:
            col = df[c].cast(pl.Utf8).cast(pl.Categorical).to_physical().cast(pl.Int32)
        feat_arrays.append(col.to_numpy().astype(np.float32))
    feature_mat_flat = np.column_stack(feat_arrays) if feat_arrays else np.empty((order_id_flat.shape[0], 0), dtype=np.float32)

    workspace_qty_flat = np.empty_like(baseline_qty_flat)

    arena = Arena(
        order_id_flat=order_id_flat,
        parent_idx_flat=parent_idx_flat,
        baseline_qty_flat=baseline_qty_flat,
        workspace_qty_flat=workspace_qty_flat,
        feature_mat_flat=feature_mat_flat,
        ds_offsets=ds_offsets,
        ds_day=ds_day,
        ds_sym_id=ds_sym_id,
        feature_names=feature_names,
    )
    return arena, oppset_index


def build_arena_from_parquet(
    parquet_glob: str | list[str],
    *,
    date_col: str = "date",
    sym_col: str = "sym",
    order_col: str = "OrderId",
    parent_col: str = "ParentId",
    root_parent_col: str = "RootParentId",
    qty_col: str = "RemainingQty",
    feature_prefix: str = "ft_",
    date_start: str | "pl.Date" | None = None,
    date_end:   str | "pl.Date" | None = None,
    strict_parent: bool = True,
    collect_streaming: bool = True,
) -> tuple[Arena, "pl.DataFrame"]:
    """
    Read Parquet(s) via Polars lazy scan, filter to date range, produce Arena.
    """
    if not HAVE_POLARS:
        raise RuntimeError("Polars not available.")

    lf = pl.scan_parquet(parquet_glob)

    required = {date_col, sym_col, order_col, parent_col, root_parent_col, qty_col}
    missing = required - set(lf.columns)
    if missing:
        raise ValueError(f"Missing required columns in Parquet: {missing}")

    feat_cols = [c for c in lf.columns if c.startswith(feature_prefix)]

    lf = lf.select(
        pl.col(date_col).cast(pl.Date).alias("date"),
        pl.col(sym_col).cast(pl.Utf8).alias("sym"),
        pl.col(order_col).cast(pl.Int64).alias("OrderId"),
        pl.col(parent_col).cast(pl.Int64).alias("ParentId"),
        pl.col(root_parent_col).cast(pl.Int64).alias("RootParentId"),
        pl.col(qty_col).cast(pl.Float64).alias("RemainingQty"),
        *[pl.col(c) for c in feat_cols],
    )
    if date_start is not None:
        lf = lf.filter(pl.col("date") >= pl.lit(date_start).cast(pl.Date))
    if date_end is not None:
        lf = lf.filter(pl.col("date") <= pl.lit(date_end).cast(pl.Date))

    df = lf.collect(streaming=collect_streaming)
    return build_arena_from_polars_df(
        df,
        date_col="date", sym_col="sym",
        order_col="OrderId", parent_col="ParentId", root_parent_col="RootParentId",
        qty_col="RemainingQty", feature_prefix=feature_prefix,
        date_start=None, date_end=None, strict_parent=strict_parent
    )


# ============================================================================
# Save / Load (NumPy bundle)
# ============================================================================

def save_arena(
    arena: Arena,
    oppset_index: "pl.DataFrame",
    out_dir: str | Path,
    *,
    persist_workspace: bool = False,
    overwrite: bool = False,
) -> dict:
    """
    Write the arena + index to disk.

    Files written (EXACTLY):
      1) oppset_index.parquet
      2) arena_core.npz
      3) features.npy
      4) workspace_init.npy   (only if persist_workspace=True)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "oppset_index": out / "oppset_index.parquet",
        "arena_core":   out / "arena_core.npz",
        "features":     out / "features.npy",
    }
    if persist_workspace:
        paths["workspace_init"] = out / "workspace_init.npy"

    for p in paths.values():
        if p.exists() and not overwrite:
            raise FileExistsError(f"{p} exists; set overwrite=True to replace.")

    # 1) oppset_index.parquet
    oppset_index.write_parquet(paths["oppset_index"])

    # 2) arena_core.npz
    np.savez_compressed(
        paths["arena_core"],
        order_id_flat=arena.order_id_flat,
        parent_idx_flat=arena.parent_idx_flat,
        baseline_qty_flat=arena.baseline_qty_flat,
        ds_offsets=arena.ds_offsets,
        ds_day=arena.ds_day,
        ds_sym_id=arena.ds_sym_id,
        feature_names=np.array([json.dumps(arena.feature_names)], dtype=object),
    )

    # 3) features.npy
    np.save(paths["features"], arena.feature_mat_flat.astype(np.float32, copy=False))

    # 4) optional workspace template
    if persist_workspace:
        np.save(paths["workspace_init"], np.zeros_like(arena.baseline_qty_flat))

    return {k: str(v) for k, v in paths.items()}


def load_arena(out_dir: str | Path, *, mmap: bool = False) -> tuple[Arena, "pl.DataFrame"]:
    """
    Load saved arena + oppset_index.
    """
    out = Path(out_dir)
    idx_path = out / "oppset_index.parquet"
    core_path = out / "arena_core.npz"
    feat_path = out / "features.npy"

    if not idx_path.exists() or not core_path.exists() or not feat_path.exists():
        raise FileNotFoundError("Missing expected files: oppset_index.parquet, arena_core.npz, features.npy")

    oppset_index = pl.read_parquet(idx_path)

    with np.load(core_path, allow_pickle=True, mmap_mode="r" if mmap else None) as core:
        order_id_flat = core["order_id_flat"].astype(np.int32, copy=False)
        parent_idx_flat = core["parent_idx_flat"].astype(np.int32, copy=False)
        baseline_qty_flat = core["baseline_qty_flat"].astype(np.float64, copy=False)
        ds_offsets = core["ds_offsets"].astype(np.int64, copy=False)
        ds_day = core["ds_day"].astype(np.int32, copy=False)
        ds_sym_id = core["ds_sym_id"].astype(np.int32, copy=False)
        feature_names = json.loads(core["feature_names"][0])

    feature_mat_flat = np.load(feat_path, mmap_mode="r" if mmap else None).astype(np.float32, copy=False)
    workspace_qty_flat = np.empty_like(baseline_qty_flat)

    arena = Arena(
        order_id_flat=order_id_flat,
        parent_idx_flat=parent_idx_flat,
        baseline_qty_flat=baseline_qty_flat,
        workspace_qty_flat=workspace_qty_flat,
        feature_mat_flat=feature_mat_flat,
        ds_offsets=ds_offsets,
        ds_day=ds_day,
        ds_sym_id=ds_sym_id,
        feature_names=feature_names,
    )
    return arena, oppset_index


# ============================================================================
# Validation (Numba)
# ============================================================================

@njit
def validate_parent_bounds(parent_idx: np.ndarray) -> bool:
    N = parent_idx.shape[0]
    for i in range(N):
        p = parent_idx[i]
        if p != ROOT and (p < 0 or p >= N):
            return False
    return True

@njit
def validate_forest(parent_idx: np.ndarray) -> bool:
    if not validate_parent_bounds(parent_idx):
        return False
    N = parent_idx.shape[0]
    state = np.zeros(N, dtype=np.uint8)  # 0=unseen,1=visiting,2=done
    for start in range(N):
        if state[start] != 0:
            continue
        node = start
        while node != ROOT and state[node] == 0:
            state[node] = 1
            node = parent_idx[node]
        if node != ROOT and node >= 0 and state[node] == 1:
            return False
        node2 = start
        while node2 != ROOT and state[node2] == 1:
            state[node2] = 2
            node2 = parent_idx[node2]
    return True

def validate_all_oppsets(arena: Arena) -> bool:
    M = arena.num_datasets()
    for m in range(M):
        s, e = arena.ds_offsets[m], arena.ds_offsets[m+1]
        if not validate_forest(arena.parent_idx_flat[s:e]):
            return False
    return True


# ============================================================================
# Fills packing + batching
# ============================================================================

def pack_fills_flat(
    per_ds_indices: Sequence[np.ndarray],  # list of (K_m,) int32
    per_ds_qtys:    Sequence[np.ndarray],  # list of (K_m,) float64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
      fills_offsets : (M+1,) int64
      fills_idx_flat: (sum K_m,) int32
      fills_qty_flat: (sum K_m,) float64
    """
    M = len(per_ds_indices)
    counts = np.fromiter((len(x) for x in per_ds_indices), count=M, dtype=np.int64)
    fills_offsets = np.zeros(M + 1, dtype=np.int64)
    np.cumsum(counts, out=fills_offsets[1:])
    if counts.sum() == 0:
        return fills_offsets, np.empty(0, np.int32), np.empty(0, np.float64)
    idx_flat = np.concatenate(per_ds_indices).astype(np.int32, copy=False)
    qty_flat = np.concatenate(per_ds_qtys).astype(np.float64, copy=False)
    return fills_offsets, idx_flat, qty_flat


def make_batches_by_ds_count(M: int, batch_size: int) -> List[np.ndarray]:
    """
    Split global oppsets [0..M-1] into contiguous batches of size <= batch_size.
    """
    batches: List[np.ndarray] = []
    start = 0
    while start < M:
        end = min(start + batch_size, M)
        batches.append(np.arange(start, end, dtype=np.int32))
        start = end
    return batches


def make_batches_by_max_rows(ds_offsets: np.ndarray, max_rows: int) -> List[np.ndarray]:
    """
    Pack oppsets into batches with approximately <= max_rows rows per batch.
    """
    M = ds_offsets.shape[0] - 1
    out: List[np.ndarray] = []
    cur: List[int] = []
    rows = 0
    for m in range(M):
        nm = int(ds_offsets[m+1] - ds_offsets[m])
        if cur and rows + nm > max_rows:
            out.append(np.array(cur, dtype=np.int32))
            cur = []
            rows = 0
        cur.append(m)
        rows += nm
    if cur:
        out.append(np.array(cur, dtype=np.int32))
    return out


# ============================================================================
# Monolithic Numba runners
# ============================================================================

@njit
def run_all_datasets_njit(ds_offsets: np.ndarray,        # (M+1,) int64
                          parent_idx_flat: np.ndarray,   # (N,) int32
                          baseline_qty_flat: np.ndarray, # (N,) float64
                          workspace_qty_flat: np.ndarray,# (N,) float64
                          fills_offsets: np.ndarray,     # (M+1,) int64
                          fills_idx_flat: np.ndarray,    # (sum K_m,) int32
                          fills_qty_flat: np.ndarray     # (sum K_m,) float64
                          ):
    """
    Run every oppset once. For oppset m:
      rows:  [ds_offsets[m]:ds_offsets[m+1])
      fills: [fills_offsets[m]:fills_offsets[m+1])
    """
    M = ds_offsets.shape[0] - 1
    for m in range(M):
        s = ds_offsets[m]
        e = ds_offsets[m+1]

        parent_local = parent_idx_flat[s:e]
        base_local   = baseline_qty_flat[s:e]
        work_local   = workspace_qty_flat[s:e]

        work_local[:] = base_local  # reset

        fs = fills_offsets[m]
        fe = fills_offsets[m+1]
        for k in range(fs, fe):
            i = fills_idx_flat[k]
            q = fills_qty_flat[k]
            work_local[i] -= q
            p = parent_local[i]
            while p != ROOT:
                work_local[p] -= q
                p = parent_local[p]


@njit
def run_subset_datasets_njit(ds_ids: np.ndarray,         # (B,) int32 global oppset ids
                             ds_offsets: np.ndarray,     # (M+1,) int64
                             parent_idx_flat: np.ndarray,
                             baseline_qty_flat: np.ndarray,
                             workspace_qty_flat: np.ndarray,
                             fills_offsets: np.ndarray,  # (M+1,) int64
                             fills_idx_flat: np.ndarray,
                             fills_qty_flat: np.ndarray):
    """
    Run only the selected oppsets (batch).
    """
    B = ds_ids.shape[0]
    for t in range(B):
        m = int(ds_ids[t])
        s = ds_offsets[m]
        e = ds_offsets[m+1]

        parent_local = parent_idx_flat[s:e]
        base_local   = baseline_qty_flat[s:e]
        work_local   = workspace_qty_flat[s:e]

        work_local[:] = base_local

        fs = fills_offsets[m]
        fe = fills_offsets[m+1]
        for k in range(fs, fe):
            i = fills_idx_flat[k]
            q = fills_qty_flat[k]
            work_local[i] -= q
            p = parent_local[i]
            while p != ROOT:
                work_local[p] -= q
                p = parent_local[p]


# ============================================================================
# Synthetic example (runs if Polars is present)
# ============================================================================

def _synthetic_polars_demo(out_dir: str = "out_demo", persist_workspace: bool = False):
    if not HAVE_POLARS:
        print("Polars not installed; skipping demo.")
        return

    # Create a tiny demo DF in your schema
    df = pl.DataFrame({
        "date": pl.date_range(low=pl.date(2025,1,1), high=pl.date(2025,1,3), interval="1d").repeat_by([3, 2, 4]).explode(),
        "sym":  (["AAPL"]*3 + ["AAPL"]*2 + ["MSFT"]*4),
        "OrderId":       [10,11,12,  20,21,  30,31,32,33],
        "ParentId":      [-1,10,10,  -1,20,  -1,30,30,31],
        "RootParentId":  [10,10,10,  20,20,  30,30,30,30],
        "RemainingQty":  [2.0,1.5,0.7, 3.0,0.5,  1.2,0.9,0.8,0.4],
        "ft_side":       ["B","S","B", "B","S", "S","S","B","B"],
        "ft_venue":      ["X","X","Y", "Y","Y", "X","Y","X","Y"],
    })

    # Build arena from eager DF (same path used by build_arena_from_parquet internally)
    arena, opp_idx = build_arena_from_polars_df(df, date_start="2025-01-01", date_end="2025-01-03")

    # Validate forests
    print("Valid forest across all oppsets? ->", validate_all_oppsets(arena))

    # Save to disk (creates EXACTLY 3 files, +1 optional)
    paths = save_arena(arena, opp_idx, out_dir, persist_workspace=persist_workspace, overwrite=True)
    print("Wrote files:")
    for k, v in paths.items():
        print("  ", k, "->", v)

    # Load back (mmap for features allowed)
    arena2, opp_idx2 = load_arena(out_dir, mmap=True)
    print("Reload check: N =", arena2.num_rows(), "M =", arena2.num_datasets(), "F =", arena2.feature_mat_flat.shape[1])

    # Build simple fills per oppset
    M = arena2.num_datasets()
    per_idx, per_qty = [], []
    for m in range(M):
        s, e = arena2.ds_offsets[m], arena2.ds_offsets[m+1]
        Nm = e - s
        if Nm >= 2:
            per_idx.append(np.array([1, 0], dtype=np.int32))
            per_qty.append(np.array([0.5, 0.25], dtype=np.float64))
        elif Nm == 1:
            per_idx.append(np.array([0], dtype=np.int32))
            per_qty.append(np.array([0.1], dtype=np.float64))
        else:
            per_idx.append(np.empty(0, np.int32))
            per_qty.append(np.empty(0, np.float64))

    fills_offsets, fills_idx_flat, fills_qty_flat = pack_fills_flat(per_idx, per_qty)

    # Run all (twice to simulate two epochs)
    run_all_datasets_njit(
        arena2.ds_offsets,
        arena2.parent_idx_flat,
        arena2.baseline_qty_flat,
        arena2.workspace_qty_flat,
        fills_offsets,
        fills_idx_flat,
        fills_qty_flat,
    )
    run_all_datasets_njit(
        arena2.ds_offsets,
        arena2.parent_idx_flat,
        arena2.baseline_qty_flat,
        arena2.workspace_qty_flat,
        fills_offsets,
        fills_idx_flat,
        fills_qty_flat,
    )

    # Inspect the first oppset
    m0 = 0
    s, e = arena2.ds_offsets[m0], arena2.ds_offsets[m0+1]
    print("Dataset 0 order_id     :", arena2.order_id_flat[s:e].tolist())
    print("Dataset 0 parent_idx   :", arena2.parent_idx_flat[s:e].tolist())
    print("Dataset 0 baseline     :", np.round(arena2.baseline_qty_flat[s:e], 3).tolist())
    print("Dataset 0 after run    :", np.round(arena2.workspace_qty_flat[s:e], 3).tolist())
    print("Feature names:", arena2.feature_names)
    print("Features shape:", arena2.feature_mat_flat.shape)


if __name__ == "__main__":
    _synthetic_polars_demo(out_dir="out_demo", persist_workspace=False)
