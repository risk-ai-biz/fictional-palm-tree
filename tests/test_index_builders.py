import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
from Indexing.oppset_batch_indexing import (
    build_global_oppset_index,
    build_batch_local_indices,
    _mix64,
)


def synthetic_data():
    oppsets = pl.DataFrame(
        {
            "date": pl.Series(
                ["2023-01-01", "2023-01-01", "2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
                dtype=pl.Date,
            ),
            "sym": ["A", "A", "B", "B", "A", "A"],
            "ping_id": [0, 1, 2, 3, 4, 5],
            "order_id": [2, 3, 4, 5, 7, 8],
            "parent_id": [1, 2, 4, 4, 6, 7],
        }
    )

    orders = pl.DataFrame(
        {
            "date": pl.Series(
                ["2023-01-01"] * 5 + ["2023-01-02"] * 3,
                dtype=pl.Date,
            ),
            "order_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "parent_id": [1, 1, 2, 4, 4, 6, 6, 7],
            "root_id": [1, 1, 1, 4, 4, 6, 6, 6],
            "level": [0, 1, 2, 0, 1, 0, 1, 2],
        }
    )

    return oppsets, orders


def test_builders_deterministic():
    oppsets, orders = synthetic_data()

    g1 = build_global_oppset_index(oppsets)
    g2 = build_global_oppset_index(oppsets)
    assert_frame_equal(g1, g2)

    b1 = build_batch_local_indices(oppsets, orders, g1)
    b2 = build_batch_local_indices(oppsets, orders, g1)

    assert_frame_equal(b1.batch_oppset_index, b2.batch_oppset_index)
    assert np.array_equal(b1.ds_row_offsets, b2.ds_row_offsets)
    assert np.array_equal(b1.ds_order_offsets, b2.ds_order_offsets)


def test_index_structures():
    oppsets, orders = synthetic_data()
    global_idx = build_global_oppset_index(oppsets)
    bundle = build_batch_local_indices(oppsets, orders, global_idx)

    # Parent map per day for ancestor checks
    orders_norm = orders.with_columns(
        pl.col("date").cast(pl.Date).cast(pl.Int32).alias("day_i32")
    )
    parent_by_day = {}
    for _, g in orders_norm.group_by("day_i32", maintain_order=True):
        d = int(g["day_i32"][0])
        parent_by_day[d] = dict(zip(g["order_id"].to_list(), g["parent_id"].to_list()))

    B = bundle.ds_order_offsets.shape[0] - 1

    # Ancestor closure
    for m in range(B):
        s, e = bundle.ds_order_offsets[m], bundle.ds_order_offsets[m + 1]
        orders_slice = bundle.order_ids_flat[s:e]
        day = int(bundle.batch_oppset_index["day_i32"][m])
        pmap = parent_by_day[day]
        order_set = set(orders_slice.tolist())
        for oid in orders_slice:
            x = int(oid)
            while True:
                pid = pmap.get(x, -1)
                if pid in (-1, x):
                    break
                assert pid in order_set
                x = pid

    # Hash table probes
    for m in range(B):
        os, oe = bundle.ds_order_offsets[m], bundle.ds_order_offsets[m + 1]
        hs, he = bundle.ht_offsets[m], bundle.ht_offsets[m + 1]
        size = he - hs
        keys = bundle.ht_keys_flat[hs:he]
        vals = bundle.ht_vals_flat[hs:he]
        mask = size - 1
        orders_slice = bundle.order_ids_flat[os:oe]
        for idx, oid in enumerate(orders_slice):
            h = int(_mix64(np.int64(oid)) & np.int64(mask))
            while True:
                v = vals[h]
                assert v != -1
                if keys[h] == oid:
                    assert v == idx
                    break
                h = (h + 1) & mask

    # Row → order mapping
    rows = bundle.rows_with_index
    for i in range(rows.height):
        m = int(rows["local_oppset_id"][i])
        oid = int(rows["order_id"][i])
        local_idx = int(bundle.row_order_local_idx[i])
        ord_start = bundle.ds_order_offsets[m]
        assert bundle.order_ids_flat[ord_start + local_idx] == oid

    # Reconstitute worker outputs
    total_orders = bundle.order_ids_flat.shape[0]
    outputs = np.empty(total_orders, dtype=np.int64)
    for m in range(B):
        s, e = bundle.ds_order_offsets[m], bundle.ds_order_offsets[m + 1]
        outputs[s:e] = np.arange(e - s, dtype=np.int64)

    records = []
    for m in range(B):
        gid = int(bundle.batch_oppset_index["global_oppset_id"][m])
        s, e = bundle.ds_order_offsets[m], bundle.ds_order_offsets[m + 1]
        orders_slice = bundle.order_ids_flat[s:e]
        vals_slice = outputs[s:e]
        for local_idx, (oid, val) in enumerate(zip(orders_slice, vals_slice)):
            assert local_idx == val
            records.append((gid, int(oid), int(val)))

    assert len(records) == total_orders
    assert len(records) == len(set(records))
