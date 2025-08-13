
"""
oppset_batch_indexing.py

Batching-aware indexing for (date, sym) oppsets with per-oppset local order indices
and tiny per-oppset hash tables for O(1) order_id→local index lookups.

Now includes lightweight **save/load helpers** so each batch writes a standard folder
you can hand directly to the compute module.

What this module does
---------------------
1) build_global_oppset_index(oppsets_df)
   → one row per (date, sym), assigns stable global_oppset_id across entire dataset

2) build_batch_local_indices(batch_oppsets_df, batch_orders_df, global_index)
   → per-batch local indices + arrays for fast access & reconstitution

3) save/load helpers:
   - save_global_index(global_index, out_dir)
   - load_global_index(path_or_dir)
   - save_batch_bundle(bundle, out_dir)
   - load_batch_bundle(out_dir, mmap=False)

Files written per batch (exactly)
---------------------------------
- batch_oppset_index.parquet
- rows_with_index.parquet
- index_arrays.npz  (contains: ds_row_offsets, ds_order_offsets, order_ids_flat,
                     parent_local_idx_flat, row_order_local_idx, ht_offsets,
                     ht_keys_flat, ht_vals_flat)

Global index file:
- global_oppset_index.parquet

No Numba or simulation logic appears here; this is purely indexing/mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Sequence, Optional
from pathlib import Path
import numpy as np

try:
    import polars as pl
except Exception as e:
    raise RuntimeError("This module requires Polars. Install polars and try again.") from e


# ----------------------------
# Small helpers
# ----------------------------

def _mix64(x: np.int64) -> np.int64:
    """64-bit mix (splitmix64-ish) for hash table slot selection."""
    z = np.uint64(x) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return z.view(np.int64)

def _next_pow2_ge(x: int) -> int:
    v = 1
    while v < x:
        v <<= 1
    return v


# ----------------------------
# Data classes
# ----------------------------

@dataclass
class BatchIndexBundle:
    batch_oppset_index: "pl.DataFrame"
    ds_row_offsets: np.ndarray
    ds_order_offsets: np.ndarray
    order_ids_flat: np.ndarray
    parent_local_idx_flat: np.ndarray
    row_order_local_idx: np.ndarray
    ht_offsets: np.ndarray
    ht_keys_flat: np.ndarray
    ht_vals_flat: np.ndarray
    rows_with_index: "pl.DataFrame"


# ----------------------------
# Global index builder
# ----------------------------

def build_global_oppset_index(oppsets_df: "pl.DataFrame") -> "pl.DataFrame":
    """
    One row per (date, sym) across the entire dataset. Assigns a stable
    global_oppset_id by sorting (day_i32, sym_id).

    Returns a Polars DataFrame with columns:
      [global_oppset_id:int64, date:pl.Date, sym:pl.Utf8, day_i32:int32, sym_id:int32, n_rows:int64]
    """
    df = oppsets_df.select(
        pl.col("date").cast(pl.Date),
        pl.col("sym").cast(pl.Utf8),
    ).with_columns(
        pl.col("date").cast(pl.Int32).alias("day_i32"),
        pl.col("sym").cast(pl.Categorical).to_physical().cast(pl.Int32).alias("sym_id"),
    )

    grp = df.group_by(["day_i32", "sym_id"]).len().sort(["day_i32", "sym_id"])
    out = grp.join(
        df.group_by(["day_i32", "sym_id"])
          .agg(pl.col("date").first().alias("date"), pl.col("sym").first().alias("sym")),
        on=["day_i32", "sym_id"], how="left"
    ).sort(["day_i32", "sym_id"]).with_columns(
        pl.int_range(0, pl.len()).cast(pl.Int64).alias("global_oppset_id")
    )[["global_oppset_id","date","sym","day_i32","sym_id","len"]].rename({"len":"n_rows"})

    return out


# ----------------------------
# Per-batch local index builder
# ----------------------------

def build_batch_local_indices(
    batch_oppsets_df: "pl.DataFrame",
    batch_orders_df: "pl.DataFrame",
    global_index: "pl.DataFrame",
) -> BatchIndexBundle:
    """
    Build per-batch local oppset/order indexing + fast lookup tables.

    - local_oppset_id: 0..B-1 for oppsets in THIS batch (sorted by (day_i32, sym_id))
    - global_oppset_id: join from global_index on (day_i32, sym_id)
    - per-oppset distinct order list including ancestors (from batch_orders_df)
    - open-addressing hash table per oppset for order_id -> local index

    Returns BatchIndexBundle.
    """
    # Normalize
    rows = batch_oppsets_df.select(
        pl.col("date").cast(pl.Date),
        pl.col("sym").cast(pl.Utf8),
        pl.col("ping_id"),
        pl.col("order_id").cast(pl.Int64),
        pl.col("parent_id").cast(pl.Int64, strict=False),
    ).with_columns(
        pl.col("date").cast(pl.Int32).alias("day_i32"),
        pl.col("sym").cast(pl.Categorical).to_physical().cast(pl.Int32).alias("sym_id"),
    ).sort(["day_i32", "sym_id"])

    orders = batch_orders_df.select(
        pl.col("date").cast(pl.Date),
        pl.col("order_id").cast(pl.Int64),
        pl.col("parent_id").cast(pl.Int64),
        pl.col("root_id").cast(pl.Int64),
        pl.col("level").cast(pl.Int32),
    ).with_columns(
        pl.col("date").cast(pl.Int32).alias("day_i32"),
    )

    # Assign local_oppset_id and local row
    rows = rows.with_columns(
        pl.row_index().over(["day_i32", "sym_id"]).cast(pl.Int32).alias("local_row")
    )
    opp_lut = (rows.group_by(["day_i32","sym_id"]).agg(pl.len().alias("n_rows"))
                    .sort(["day_i32","sym_id"])
                    .with_columns(pl.int_range(0, pl.len()).cast(pl.Int32).alias("local_oppset_id")))

    # Join global_oppset_id
    key = global_index.select(["day_i32","sym_id","global_oppset_id"]).unique().sort(["day_i32","sym_id"])
    opp_lut = opp_lut.join(key, on=["day_i32","sym_id"], how="left")
    if opp_lut.select(pl.col("global_oppset_id").is_null().any()).item():
        raise ValueError("Some batch (date,sym) not found in global_index. Did you build global over full dataset?")

    rows = rows.join(opp_lut, on=["day_i32","sym_id"], how="left").with_row_index("batch_row")

    # Per-day parent map
    parent_by_day: Dict[int, Dict[int,int]] = {}
    level_by_day:  Dict[int, Dict[int,int]] = {}
    for _, g in orders.group_by("day_i32", maintain_order=True):
        d = int(g["day_i32"][0])
        parent_by_day[d] = dict(zip(g["order_id"].to_list(), g["parent_id"].to_list()))
        level_by_day[d] = dict(zip(g["order_id"].to_list(), g["level"].to_list()))

    # Build row offsets (per oppset)
    grp = rows.group_by(["day_i32","sym_id"]).len().sort(["day_i32","sym_id"])
    counts = grp["len"].to_numpy().astype(np.int64)
    ds_row_offsets = np.zeros(counts.shape[0] + 1, dtype=np.int64)
    np.cumsum(counts, out=ds_row_offsets[1:])

    # Pre-materialize arrays
    rows_day = rows["day_i32"].to_numpy().astype(np.int32, copy=False)
    rows_oid = rows["order_id"].to_numpy().astype(np.int64, copy=False)

    # Iterate oppsets in local order
    B = counts.shape[0]
    starts = ds_row_offsets[:-1]; ends = ds_row_offsets[1:]
    opp_day = grp["day_i32"].to_numpy().astype(np.int32, copy=False)

    ds_order_offsets = np.zeros(B + 1, dtype=np.int64)
    order_ids_chunks: List[np.ndarray] = []
    parent_local_chunks: List[np.ndarray] = []
    ht_offsets = np.zeros(B + 1, dtype=np.int64)
    ht_keys_chunks: List[np.ndarray] = []
    ht_vals_chunks: List[np.ndarray] = []
    row_order_local_idx = np.empty(rows.height, dtype=np.int32)

    for m in range(B):
        s, e = int(starts[m]), int(ends[m])
        d = int(opp_day[m])
        # Unique referenced child oids
        uniq_oids = np.unique(rows_oid[s:e])

        # Closure with ancestors
        pmap = parent_by_day.get(d, {})
        lmap = level_by_day.get(d, {})

        order_set: Dict[int,None] = {}

        def add_with_ancestors(oid: int):
            x = int(oid)
            visited = 0
            while True:
                if x in order_set:
                    return
                order_set[x] = None
                if x not in pmap:
                    break
                px = pmap[x]
                if px == x or px < 0:
                    break
                x = px
                visited += 1
                if visited > 1_000_000:
                    raise RuntimeError("Ancestor loop too deep (cycle?)")

        for oid in uniq_oids:
            add_with_ancestors(int(oid))

        orders_list = list(order_set.keys())
        orders_list.sort(key=lambda x: (lmap.get(x, 1_000_000), x))   # by level then id
        n_orders = len(orders_list)
        local_index = {orders_list[i]: i for i in range(n_orders)}

        # parent local
        parent_local = np.empty(n_orders, dtype=np.int32)
        for i, oid in enumerate(orders_list):
            pid = pmap.get(oid, -1)
            if pid == oid or pid < 0:
                parent_local[i] = -1
            else:
                parent_local[i] = np.int32(local_index.get(pid, -1))

        # Build small hash table (order_id -> local)
        if n_orders == 0:
            ht_size = 1
            keys = np.empty(1, dtype=np.int64)
            vals = np.full(1, -1, dtype=np.int32)
        else:
            ht_size = _next_pow2_ge(max(4, n_orders * 2))
            keys = np.empty(ht_size, dtype=np.int64)
            vals = np.full(ht_size, -1, dtype=np.int32)
            mask = ht_size - 1
            for i, oid in enumerate(orders_list):
                h = int(_mix64(np.int64(oid)) & np.int64(mask))
                while vals[h] != -1:
                    h = (h + 1) & mask
                keys[h] = np.int64(oid)
                vals[h] = np.int32(i)

        # Map rows to local order idx
        mask = ht_size - 1
        for idx in range(s, e):
            oid = np.int64(rows_oid[idx])
            h = int(_mix64(oid) & np.int64(mask))
            while True:
                v = vals[h]
                if v == -1:
                    raise KeyError(f"order_id {int(oid)} not found in oppset m={m}")
                if keys[h] == oid:
                    row_order_local_idx[idx] = v
                    break
                h = (h + 1) & mask

        order_ids_chunks.append(np.array(orders_list, dtype=np.int64))
        parent_local_chunks.append(parent_local)
        ds_order_offsets[m+1] = ds_order_offsets[m] + n_orders

        ht_keys_chunks.append(keys)
        ht_vals_chunks.append(vals)
        ht_offsets[m+1] = ht_offsets[m] + ht_size

    order_ids_flat = np.concatenate(order_ids_chunks, axis=0) if order_ids_chunks else np.empty(0, np.int64)
    parent_local_idx_flat = np.concatenate(parent_local_chunks, axis=0) if parent_local_chunks else np.empty(0, np.int32)
    ht_keys_flat = np.concatenate(ht_keys_chunks, axis=0) if ht_keys_chunks else np.empty(0, np.int64)
    ht_vals_flat = np.concatenate(ht_vals_chunks, axis=0) if ht_vals_chunks else np.empty(0, np.int32)

    # Build batch_oppset_index
    base = rows.group_by(["day_i32","sym_id"]).agg(
        pl.col("date").first().alias("date"),
        pl.col("sym").first().alias("sym"),
        pl.len().alias("n_rows"),
        pl.col("local_oppset_id").first().alias("local_oppset_id"),
        pl.col("global_oppset_id").first().alias("global_oppset_id"),
    ).sort(["day_i32","sym_id"])

    batch_oppset_index = base.with_columns([
        pl.Series("row_start", ds_row_offsets[:-1]),
        pl.Series("row_end", ds_row_offsets[1:]),
        pl.Series("ord_start", ds_order_offsets[:-1]),
        pl.Series("ord_end", ds_order_offsets[1:]),
        pl.Series("n_orders", ds_order_offsets[1:] - ds_order_offsets[:-1]),
        pl.Series("ht_start", ht_offsets[:-1]),
        pl.Series("ht_size", ht_offsets[1:] - ht_offsets[:-1]),
    ])[["local_oppset_id","global_oppset_id","date","sym","day_i32","sym_id",
        "row_start","row_end","n_rows","ord_start","ord_end","n_orders","ht_start","ht_size"]]

    # rows_with_index (lightweight)
    rows_with_index = rows.select(
        "date","sym","ping_id","order_id",
        "day_i32","sym_id","local_oppset_id","global_oppset_id",
        "local_row","batch_row"
    )

    return BatchIndexBundle(
        batch_oppset_index=batch_oppset_index,
        ds_row_offsets=ds_row_offsets,
        ds_order_offsets=ds_order_offsets,
        order_ids_flat=order_ids_flat,
        parent_local_idx_flat=parent_local_idx_flat,
        row_order_local_idx=row_order_local_idx,
        ht_offsets=ht_offsets,
        ht_keys_flat=ht_keys_flat,
        ht_vals_flat=ht_vals_flat,
        rows_with_index=rows_with_index,
    )


# ----------------------------
# Save / Load helpers
# ----------------------------

def save_global_index(global_index: "pl.DataFrame", out_dir: str | Path, *, overwrite: bool=False) -> str:
    """
    Write the global index to `<out_dir>/global_oppset_index.parquet`.
    Returns the file path.
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / "global_oppset_index.parquet"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; set overwrite=True to replace.")
    global_index.write_parquet(path)
    return str(path)


def load_global_index(path_or_dir: str | Path) -> "pl.DataFrame":
    """
    Load `global_oppset_index.parquet`. If given a directory, append the filename.
    """
    p = Path(path_or_dir)
    if p.is_dir():
        p = p / "global_oppset_index.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Global index not found at {p}")
    return pl.read_parquet(p)


def save_batch_bundle(bundle: BatchIndexBundle, out_dir: str | Path, *, overwrite: bool=False, compress: bool=True) -> dict:
    """
    Persist a batch bundle to `<out_dir>` with exactly three files:

      - batch_oppset_index.parquet
      - rows_with_index.parquet
      - index_arrays.npz

    Returns a dict of paths.
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    paths = {
        "batch_oppset_index": out / "batch_oppset_index.parquet",
        "rows_with_index": out / "rows_with_index.parquet",
        "arrays": out / "index_arrays.npz",
    }
    for p in paths.values():
        if p.exists() and not overwrite:
            raise FileExistsError(f"{p} exists; set overwrite=True to replace.")

    # DataFrames
    bundle.batch_oppset_index.write_parquet(paths["batch_oppset_index"])
    bundle.rows_with_index.write_parquet(paths["rows_with_index"])

    # Arrays
    arrays = dict(
        ds_row_offsets=bundle.ds_row_offsets,
        ds_order_offsets=bundle.ds_order_offsets,
        order_ids_flat=bundle.order_ids_flat,
        parent_local_idx_flat=bundle.parent_local_idx_flat,
        row_order_local_idx=bundle.row_order_local_idx,
        ht_offsets=bundle.ht_offsets,
        ht_keys_flat=bundle.ht_keys_flat,
        ht_vals_flat=bundle.ht_vals_flat,
    )
    if compress:
        np.savez_compressed(paths["arrays"], **arrays)
    else:
        np.savez(paths["arrays"], **arrays)

    return {k: str(v) for k, v in paths.items()}


def load_batch_bundle(out_dir: str | Path, *, mmap: bool=False) -> BatchIndexBundle:
    """
    Read a persisted batch bundle from `<out_dir>` and return a BatchIndexBundle.
    """
    out = Path(out_dir)
    p_index = out / "batch_oppset_index.parquet"
    p_rows  = out / "rows_with_index.parquet"
    p_npz   = out / "index_arrays.npz"

    if not (p_index.exists() and p_rows.exists() and p_npz.exists()):
        raise FileNotFoundError(f"Missing files. Expected: {p_index}, {p_rows}, {p_npz}")

    batch_oppset_index = pl.read_parquet(p_index)
    rows_with_index = pl.read_parquet(p_rows)

    with np.load(p_npz, allow_pickle=False, mmap_mode=("r" if mmap else None)) as z:
        ds_row_offsets = z["ds_row_offsets"].astype(np.int64, copy=False)
        ds_order_offsets = z["ds_order_offsets"].astype(np.int64, copy=False)
        order_ids_flat = z["order_ids_flat"].astype(np.int64, copy=False)
        parent_local_idx_flat = z["parent_local_idx_flat"].astype(np.int32, copy=False)
        row_order_local_idx = z["row_order_local_idx"].astype(np.int32, copy=False)
        ht_offsets = z["ht_offsets"].astype(np.int64, copy=False)
        ht_keys_flat = z["ht_keys_flat"].astype(np.int64, copy=False)
        ht_vals_flat = z["ht_vals_flat"].astype(np.int32, copy=False)

    return BatchIndexBundle(
        batch_oppset_index=batch_oppset_index,
        ds_row_offsets=ds_row_offsets,
        ds_order_offsets=ds_order_offsets,
        order_ids_flat=order_ids_flat,
        parent_local_idx_flat=parent_local_idx_flat,
        row_order_local_idx=row_order_local_idx,
        ht_offsets=ht_offsets,
        ht_keys_flat=ht_keys_flat,
        ht_vals_flat=ht_vals_flat,
        rows_with_index=rows_with_index,
    )


# ----------------------------
# Demo
# ----------------------------

def _demo():
    # Global oppsets (union)
    oppsets = pl.DataFrame({
        "date": [pl.date(2025,1,1)]*4 + [pl.date(2025,1,2)]*3,
        "sym":  ["AAPL"]*4 + ["MSFT"]*3,
        "ping_id": [{"date":pl.date(2025,1,1),"order_id":100,"sequence_no":1},
                    {"date":pl.date(2025,1,1),"order_id":101,"sequence_no":2},
                    {"date":pl.date(2025,1,1),"order_id":103,"sequence_no":3},
                    {"date":pl.date(2025,1,1),"order_id":104,"sequence_no":4},
                    {"date":pl.date(2025,1,2),"order_id":200,"sequence_no":1},
                    {"date":pl.date(2025,1,2),"order_id":201,"sequence_no":2},
                    {"date":pl.date(2025,1,2),"order_id":203,"sequence_no":3}],
        "order_id": [100,101,103,104, 200,201,203],
        "parent_id":[-1,100,101,101, -1,200,201],
    })

    orders = pl.DataFrame({
        "date": [pl.date(2025,1,1)]*5 + [pl.date(2025,1,2)]*4,
        "order_id": [100,101,102,103,104, 200,201,202,203],
        "parent_id":[-1,100,100,101,101, -1,200,200,201],
        "root_id":  [100,100,100,100,100, 200,200,200,200],
        "level":    [0,1,1,2,2, 0,1,1,2],
    })

    # 1) Global index
    global_idx = build_global_oppset_index(oppsets)

    # 2) Single batch over all oppsets
    bundle = build_batch_local_indices(oppsets, orders, global_idx)

    # 3) Save
    out = Path("batch_demo_out"); out.mkdir(exist_ok=True, parents=True)
    gpath = save_global_index(global_idx, out, overwrite=True)
    bpaths = save_batch_bundle(bundle, out / "batch_0001", overwrite=True)

    # 4) Load
    g2 = load_global_index(out)  # dir or file OK
    b2 = load_batch_bundle(out / "batch_0001", mmap=True)

    print("Saved global index ->", gpath)
    print("Saved batch files  ->", bpaths)
    print("Reload check: B rows", b2.rows_with_index.height, "| orders_flat", b2.order_ids_flat.shape)

if __name__ == "__main__":
    _demo()
