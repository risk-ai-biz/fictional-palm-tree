"""
Numba kernels for oppset computations.

This module provides low-level ``@njit`` functions that traverse the
flattened oppset representation produced by :mod:`oppset_batch_indexing`.

Functions follow the Numba checklist:
  * arrays must be C-contiguous ``np.ndarray`` instances
  * no Python objects appear inside hot loops
  * outputs are preallocated and written in-place
  * optional debug assertions may be enabled via ``DEBUG = True``

The main kernels iterate oppsets using ``ds_row_offsets`` and
``ds_order_offsets``, map rows to local order indices via
``row_order_local_idx``, and climb ancestor links with
``parent_local_idx_flat``.

A small Python driver ``compute_batch_roots_and_depths`` demonstrates how
these kernels can be applied per batch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, List, Dict

import numpy as np
from numba import njit

from .oppset_batch_indexing import load_batch_bundle

# Enable or disable debug assertions within the kernels.
DEBUG = False


@njit(cache=True)
def row_root_local_idx(
    ds_row_offsets: np.ndarray,
    ds_order_offsets: np.ndarray,
    row_order_local_idx: np.ndarray,
    parent_local_idx_flat: np.ndarray,
    out_root: np.ndarray,
) -> None:
    """Compute root local order index for each row.

    Parameters are 1-D ``np.ndarray`` instances that must be C-contiguous.
    ``out_root`` must be preallocated to ``row_order_local_idx.shape[0]``.
    """
    if DEBUG:
        # C-contiguity checks (1-D arrays)
        assert ds_row_offsets.strides[0] == ds_row_offsets.dtype.itemsize
        assert ds_order_offsets.strides[0] == ds_order_offsets.dtype.itemsize
        assert row_order_local_idx.strides[0] == row_order_local_idx.dtype.itemsize
        assert parent_local_idx_flat.strides[0] == parent_local_idx_flat.dtype.itemsize
        assert out_root.strides[0] == out_root.dtype.itemsize

    B = ds_row_offsets.shape[0] - 1
    for b in range(B):
        row_start = ds_row_offsets[b]
        row_end = ds_row_offsets[b + 1]
        ord_start = ds_order_offsets[b]
        for ridx in range(row_start, row_end):
            local_idx = row_order_local_idx[ridx]
            parent = parent_local_idx_flat[ord_start + local_idx]
            while parent != -1:
                local_idx = parent
                parent = parent_local_idx_flat[ord_start + local_idx]
            out_root[ridx] = local_idx


@njit(cache=True)
def row_depth_from_root(
    ds_row_offsets: np.ndarray,
    ds_order_offsets: np.ndarray,
    row_order_local_idx: np.ndarray,
    parent_local_idx_flat: np.ndarray,
    out_depth: np.ndarray,
) -> None:
    """Compute depth of each row's order from its root."""
    if DEBUG:
        assert ds_row_offsets.strides[0] == ds_row_offsets.dtype.itemsize
        assert ds_order_offsets.strides[0] == ds_order_offsets.dtype.itemsize
        assert row_order_local_idx.strides[0] == row_order_local_idx.dtype.itemsize
        assert parent_local_idx_flat.strides[0] == parent_local_idx_flat.dtype.itemsize
        assert out_depth.strides[0] == out_depth.dtype.itemsize

    B = ds_row_offsets.shape[0] - 1
    for b in range(B):
        row_start = ds_row_offsets[b]
        row_end = ds_row_offsets[b + 1]
        ord_start = ds_order_offsets[b]
        for ridx in range(row_start, row_end):
            local_idx = row_order_local_idx[ridx]
            depth = 0
            parent = parent_local_idx_flat[ord_start + local_idx]
            while parent != -1:
                depth += 1
                local_idx = parent
                parent = parent_local_idx_flat[ord_start + local_idx]
            out_depth[ridx] = depth


def compute_batch_roots_and_depths(batch_dirs: Sequence[str | Path]) -> List[Dict[str, np.ndarray]]:
    """Driver function that runs kernels over multiple batches.

    Parameters
    ----------
    batch_dirs:
        Iterable of directories containing saved ``BatchIndexBundle``
        artifacts (see :func:`oppset_batch_indexing.load_batch_bundle`).

    Returns
    -------
    list of dict
        One entry per batch with keys ``"batch_dir"``, ``"root_local_idx"``
        and ``"depth"`` containing NumPy arrays.
    """
    outputs: List[Dict[str, np.ndarray]] = []
    for bd in batch_dirs:
        bundle = load_batch_bundle(bd, mmap=True)
        ds_row_offsets = np.ascontiguousarray(bundle.ds_row_offsets)
        ds_order_offsets = np.ascontiguousarray(bundle.ds_order_offsets)
        row_order_local_idx = np.ascontiguousarray(bundle.row_order_local_idx)
        parent_local_idx_flat = np.ascontiguousarray(bundle.parent_local_idx_flat)

        n_rows = row_order_local_idx.shape[0]
        root_local = np.empty(n_rows, dtype=np.int32)
        depth = np.empty(n_rows, dtype=np.int32)

        row_root_local_idx(ds_row_offsets, ds_order_offsets,
                            row_order_local_idx, parent_local_idx_flat,
                            root_local)
        row_depth_from_root(ds_row_offsets, ds_order_offsets,
                             row_order_local_idx, parent_local_idx_flat,
                             depth)

        outputs.append({
            "batch_dir": str(bd),
            "root_local_idx": root_local,
            "depth": depth,
        })
    return outputs


__all__ = [
    "row_root_local_idx",
    "row_depth_from_root",
    "compute_batch_roots_and_depths",
]
