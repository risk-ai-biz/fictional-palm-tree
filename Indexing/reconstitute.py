from __future__ import annotations

"""Utilities to map worker outputs back to global identifiers."""

from typing import Sequence, Tuple

import numpy as np
try:
    import polars as pl
except Exception as e:  # pragma: no cover - polars is required
    raise RuntimeError("This module requires Polars. Install polars and try again.") from e


def reconstitute_outputs(
    bundle,
    local_pairs: Sequence[Tuple[int, int]] | np.ndarray,
    values: Sequence[float] | np.ndarray,
) -> "pl.DataFrame":
    """Rebuild a DataFrame keyed by global ids.

    Parameters
    ----------
    bundle:
        ``BatchIndexBundle`` produced by ``build_batch_local_indices``.
    local_pairs:
        Iterable of ``(local_oppset_id, local_order_idx)`` pairs.
    values:
        Iterable of values aligned with ``local_pairs``.

    Returns
    -------
    pl.DataFrame
        Columns ``[date, sym, global_oppset_id, order_id, value]``.
    """
    pairs = np.asarray(local_pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("local_pairs must be of shape (N, 2)")

    vals = np.asarray(values)
    if vals.shape[0] != pairs.shape[0]:
        raise ValueError("values must align with local_pairs")

    local_oppset_ids = pairs[:, 0].astype(np.int64)
    local_order_idxs = pairs[:, 1].astype(np.int64)

    # Map local oppset ids to global ids
    oppset_map = (
        bundle.batch_oppset_index.sort("local_oppset_id")
        ["global_oppset_id"].to_numpy().astype(np.int64)
    )
    global_oppset_ids = oppset_map[local_oppset_ids]

    # Map local order index within each oppset to global order_id
    start_idx = bundle.ds_order_offsets[local_oppset_ids] + local_order_idxs
    order_ids = bundle.order_ids_flat[start_idx]

    df = pl.DataFrame(
        {
            "global_oppset_id": global_oppset_ids,
            "order_id": order_ids,
            "value": vals,
        }
    )

    enriched = df.join(
        bundle.batch_oppset_index.select(["global_oppset_id", "date", "sym"]),
        on="global_oppset_id",
        how="left",
    )[["date", "sym", "global_oppset_id", "order_id", "value"]]

    return enriched
