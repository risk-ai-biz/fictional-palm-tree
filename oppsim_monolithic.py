
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        def deco(fn): return fn
        if args and callable(args[0]): return args[0]
        return deco

ROOT = np.int32(-1)

@dataclass
class Arena:
    order_id_flat:      np.ndarray  # (N,) int32
    parent_idx_flat:    np.ndarray  # (N,) int32
    baseline_qty_flat:  np.ndarray  # (N,) float64
    workspace_qty_flat: np.ndarray  # (N,) float64
    ds_offsets: np.ndarray          # (M+1,) int64
    ds_day:     np.ndarray          # (M,)   int32
    ds_sym_id:  np.ndarray          # (M,)   int32
    def num_rows(self) -> int: return int(self.order_id_flat.shape[0])
    def num_datasets(self) -> int: return int(self.ds_offsets.shape[0] - 1)

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

def build_arena_from_numpy(
    day: np.ndarray,             # (Nraw,) int32
    sym_id: np.ndarray,          # (Nraw,) int32
    order_id: np.ndarray,        # (Nraw,) int32
    parent_order_id: np.ndarray, # (Nraw,) int32 (-1 or valid order_id within group)
    baseline_qty: np.ndarray,    # (Nraw,) float64
) -> Arena:
    day = np.ascontiguousarray(day, dtype=np.int32)
    sym_id = np.ascontiguousarray(sym_id, dtype=np.int32)
    order_id = np.ascontiguousarray(order_id, dtype=np.int32)
    parent_order_id = np.ascontiguousarray(parent_order_id, dtype=np.int32)
    baseline_qty = np.ascontiguousarray(baseline_qty, dtype=np.float64)

    Nraw = day.shape[0]
    assert sym_id.shape[0] == Nraw == order_id.shape[0] == parent_order_id.shape[0] == baseline_qty.shape[0]

    key = day.astype(np.int64) * (1 << 33) + sym_id.astype(np.int64) * (1 << 22) + order_id.astype(np.int64)
    sort_idx = np.argsort(key, kind="mergesort")
    day_s   = day[sort_idx]
    sym_s   = sym_id[sort_idx]
    oid_s   = order_id[sort_idx]
    poid_s  = parent_order_id[sort_idx]
    qty_s   = baseline_qty[sort_idx]

    ds_keys = (day_s.astype(np.int64) << 32) | (sym_s.astype(np.int64) & 0xFFFFFFFF)
    change = np.ones(Nraw, dtype=bool)
    change[1:] = ds_keys[1:] != ds_keys[:-1]
    ds_starts = np.nonzero(change)[0]
    ds_offsets = np.empty(ds_starts.shape[0] + 1, dtype=np.int64)
    ds_offsets[:-1] = ds_starts
    ds_offsets[-1] = Nraw
    ds_day = day_s[ds_starts]
    ds_sym_id = sym_s[ds_starts]

    parent_idx_flat = np.empty(Nraw, dtype=np.int32)
    for m in range(ds_offsets.shape[0] - 1):
        s = ds_offsets[m]; e = ds_offsets[m+1]
        Nm = e - s
        idx_map = {int(oid_s[s+i]): np.int32(i) for i in range(Nm)}
        for i in range(Nm):
            p_oid = int(poid_s[s+i])
            if p_oid == -1:
                parent_idx_flat[s+i] = ROOT
            else:
                if p_oid not in idx_map:
                    raise ValueError(f"Missing parent_order_id {p_oid} in dataset (day={int(ds_day[m])}, sym={int(ds_sym_id[m])}).")
                parent_idx_flat[s+i] = idx_map[p_oid]

    workspace_qty_flat = np.empty_like(qty_s)
    return Arena(
        order_id_flat=oid_s,
        parent_idx_flat=parent_idx_flat,
        baseline_qty_flat=qty_s,
        workspace_qty_flat=workspace_qty_flat,
        ds_offsets=ds_offsets,
        ds_day=ds_day,
        ds_sym_id=ds_sym_id,
    )

@njit
def run_one_dataset(ds_id: int,
                    ds_offsets: np.ndarray,        # (M+1,) int64
                    parent_idx_flat: np.ndarray,   # (N,) int32
                    baseline_qty_flat: np.ndarray, # (N,) float64
                    workspace_qty_flat: np.ndarray,# (N,) float64
                    fill_indices: np.ndarray,      # (K,) int32
                    fill_qtys: np.ndarray          # (K,) float64
                    ):
    start = ds_offsets[ds_id]
    end   = ds_offsets[ds_id + 1]
    parent_idx_local = parent_idx_flat[start:end]
    baseline_local   = baseline_qty_flat[start:end]
    workspace_local  = workspace_qty_flat[start:end]
    workspace_local[:] = baseline_local
    K = fill_indices.shape[0]
    for k in range(K):
        i = fill_indices[k]
        q = fill_qtys[k]
        workspace_local[i] -= q
        p = parent_idx_local[i]
        while p != -1:
            workspace_local[p] -= q
            p = parent_idx_local[p]

@njit
def run_all_datasets(ds_offsets: np.ndarray,        # (M+1,) int64
                     parent_idx_flat: np.ndarray,   # (N,) int32
                     baseline_qty_flat: np.ndarray, # (N,) float64
                     workspace_qty_flat: np.ndarray,# (N,) float64
                     fills_offsets: np.ndarray,     # (M+1,) int64
                     fills_idx_flat: np.ndarray,    # (sumK,) int32
                     fills_qty_flat: np.ndarray     # (sumK,) float64
                     ):
    M = ds_offsets.shape[0] - 1
    for m in range(M):
        start = ds_offsets[m]
        end   = ds_offsets[m+1]
        parent_idx_local = parent_idx_flat[start:end]
        baseline_local   = baseline_qty_flat[start:end]
        workspace_local  = workspace_qty_flat[start:end]
        workspace_local[:] = baseline_local
        fs = fills_offsets[m]; fe = fills_offsets[m+1]
        for k in range(fs, fe):
            i = fills_idx_flat[k]
            q = fills_qty_flat[k]
            workspace_local[i] -= q
            p = parent_idx_local[i]
            while p != -1:
                workspace_local[p] -= q
                p = parent_idx_local[p]

# --- small demo ---
if __name__ == "__main__":
    day   = np.array([0,0,0,0,  0,0,  1,1,1], dtype=np.int32)
    sym   = np.array([0,0,0,0,  1,1,  0,0,1], dtype=np.int32)
    oid   = np.array([10,11,12,13,  20,21,  30,31,  40], dtype=np.int32)
    poid  = np.array([-1,10,10,11,  -1,20,  -1,30,  -1], dtype=np.int32)
    qty   = np.array([2.0,1.5,0.7,1.0,  3.0,0.5,  1.2,0.9,  4.0], dtype=np.float64)
    arena = build_arena_from_numpy(day, sym, oid, poid, qty)

    # Validate
    ok_all = True
    for m in range(arena.num_datasets()):
        s = arena.ds_offsets[m]; e = arena.ds_offsets[m+1]
        ok_all = ok_all and validate_forest(arena.parent_idx_flat[s:e])
    print("Forest valid across demo datasets?:", bool(ok_all))

    # Build simple fills
    M = arena.num_datasets()
    fill_chunks_idx: List[np.ndarray] = []
    fill_chunks_qty: List[np.ndarray] = []
    for m in range(M):
        s = arena.ds_offsets[m]; e = arena.ds_offsets[m+1]
        Nm = e - s
        if Nm >= 2:
            fill_chunks_idx.append(np.array([1,0], dtype=np.int32))
            fill_chunks_qty.append(np.array([0.5,0.25], dtype=np.float64))
        elif Nm == 1:
            fill_chunks_idx.append(np.array([0], dtype=np.int32))
            fill_chunks_qty.append(np.array([0.1], dtype=np.float64))
        else:
            fill_chunks_idx.append(np.empty(0, dtype=np.int32))
            fill_chunks_qty.append(np.empty(0, dtype=np.float64))

    counts = np.array([arr.shape[0] for arr in fill_chunks_idx], dtype=np.int64)
    fills_offsets = np.zeros(M + 1, dtype=np.int64); np.cumsum(counts, out=fills_offsets[1:])
    fills_idx_flat = np.concatenate(fill_chunks_idx, dtype=np.int32) if counts.sum() else np.empty(0, np.int32)
    fills_qty_flat = np.concatenate(fill_chunks_qty, dtype=np.float64) if counts.sum() else np.empty(0, np.float64)

    # Run twice
    run_all_datasets(
        arena.ds_offsets,
        arena.parent_idx_flat,
        arena.baseline_qty_flat,
        arena.workspace_qty_flat,
        fills_offsets,
        fills_idx_flat,
        fills_qty_flat,
    )
    run_all_datasets(
        arena.ds_offsets,
        arena.parent_idx_flat,
        arena.baseline_qty_flat,
        arena.workspace_qty_flat,
        fills_offsets,
        fills_idx_flat,
        fills_qty_flat,
    )

    # Inspect first dataset
    m = 0
    s = arena.ds_offsets[m]; e = arena.ds_offsets[m+1]
    print("Dataset 0 baseline:", np.round(arena.baseline_qty_flat[s:e], 3))
    print("Dataset 0 result  :", np.round(arena.workspace_qty_flat[s:e], 3))
