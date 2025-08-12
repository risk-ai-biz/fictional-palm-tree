
# Oppset / Order Indexing — Implementation Spec (Polars → Batches → Numba-ready)

**Scope**: This document fully specifies the **indexing/mapping** layer only.  
It covers how to construct global and per-batch indices with Polars and how another
agent can later implement Numba-based inner/outer simulation loops that consume
these indices. No simulation math is included here.

Companion files:
- `oppset_batch_indexing.py` — lean, batching-aware indexing builders.
- `oppset_mapping.svg` — visual overview of the mappings (global ↔ batch).
  
---

## 1) Purpose

We split a large dataset into **N batches** of oppsets (each batch written to its own Parquet files). We need:

1. A **global oppset index** over the entire dataset so we can reassemble results after parallel processing.
2. For each batch, a **local index** that gives workers cache-friendly structures:
   - contiguous **row slices** per oppset,
   - contiguous **order lists** per oppset (0..n_orders-1) **including ancestors**, and
   - tiny **hash tables** so `order_id → local_order_idx` is O(1).

Workers compute results using **local ids**; the driver maps them back using the **global index**.

---

## 2) Notation & Types

- An **oppset** is identified by `(date, sym)`; in code we encode this as `(day_i32, sym_id)`.
- `day_i32 : int32` — days since epoch; `sym_id : int32` — categorical physical code of `sym`.
- **Global oppset id**: `global_oppset_id : int64` — stable across the entire dataset (sorted by `(day_i32, sym_id)`).
- **Local oppset id**: `local_oppset_id : int32` — stable **within a batch** (sorted by `(day_i32, sym_id)` inside that batch).
- `order_id : int64` — unique per day; `parent_id : int64`; `root_id : int64`; `level : int32` (0 == root).
- Arrays are **C-contiguous**; integer dtypes are **int32** or **int64** as specified for Numba friendliness.

---

## 3) Input Schemas

### Oppsets (per batch and global-union)
- `date : pl.Date`
- `sym : pl.Utf8`
- `ping_id : pl.Struct({...})` — unique row key; not parsed by the indexers
- `order_id : pl.Int64`
- `parent_id : pl.Int64` *(optional for indexing; ancestry comes from Orders)*

### Orders (per batch and global-union)
- `date : pl.Date`
- `order_id : pl.Int64`
- `parent_id : pl.Int64`
- `root_id : pl.Int64`
- `level : pl.Int32` *(0 == root)*

**Assumptions**
- `order_id` uniqueness is **per day**.
- The full ancestor chain exists in the **Orders** table for each `(date, order_id)` we may see in oppsets.

---

## 4) GlobalOppsetIndex (built once over the full dataset)

**Purpose**: Assign a stable `global_oppset_id` for every `(date, sym)` in the **union** of all oppsets.

### Algorithm (Polars)

1. Cast:
   - `day_i32 = date.cast(Int32)`
   - `sym_id = sym.cast(Categorical).to_physical().cast(Int32)`
2. Group: `group_by([day_i32, sym_id]).len()` → `n_rows` per oppset.
3. Sort by `day_i32, sym_id`.
4. Assign `global_oppset_id = int_range(0, len)`.
5. Join back `date, sym` (first-of-group) for convenience.

### Output (DataFrame: `GlobalOppsetIndex`)

| column            | dtype  | description                                 |
|-------------------|--------|---------------------------------------------|
| global_oppset_id  | int64  | stable ID across the entire dataset         |
| date              | Date   |                                            |
| sym               | Utf8   |                                            |
| day_i32           | int32  | `date` as days since epoch                  |
| sym_id            | int32  | categorical code for `sym`                  |
| n_rows            | int64  | total rows in this oppset                   |

**Determinism**: Sorting strictly by `(day_i32, sym_id)` ensures stable IDs regardless of input file order.

---

## 5) BatchLocalIndices (built per batch)

**Purpose**: Give each worker compact, cache-friendly structures for **its** subset of oppsets.

### Step A — Normalize and tag rows

For the batch’s oppsets table:
- Cast `date → day_i32:int32` and `sym → sym_id:int32`.
- Sort by `(day_i32, sym_id)` to pack each oppset’s rows contiguously.
- Compute `local_row:int32 = cum_count() over [day_i32, sym_id]`.
- Build `opp_lut = group_by([day_i32, sym_id]).agg(n_rows).sort(...).with_columns(local_oppset_id = int_range(...))`.
- Join `global_oppset_id` from `GlobalOppsetIndex` on `(day_i32, sym_id)`.
- Add `batch_row = with_row_index()`.

**Row slices per oppset**:
- `ds_row_offsets : int64 (B+1)` — `rows[s:e]` belongs to `local_oppset_id = m`.
  - Build via cumulative sum of `n_rows` from the grouped order.

### Step B — Per-day parent & level maps

From the batch’s **Orders** table:
- Normalize `date → day_i32`.
- Build two Python dicts **per day**:
  - `parent_by_day[day_i32][order_id] = parent_id`
  - `level_by_day[day_i32][order_id]  = level`

*(This is fast and makes per-oppset closure computation trivial.)*

### Step C — Per-oppset distinct order lists (local 0..n_orders-1)

For each local oppset `m` (in increasing `(day_i32, sym_id)` order):

1. Get the **unique** `order_id` values appearing in its row slice `rows[s:e]`.
2. For each such `order_id`, compute the **ancestor closure** using `parent_by_day[day]`:
   - add `order_id` to a set;
   - iteratively replace with `parent_id` until `parent == id` or `parent < 0` or `parent` missing;
   - guard with a ceiling on steps (`> 1e6` → cycle suspicion → raise).
3. Convert the set to a list and **sort** by `(level asc, order_id asc)`, with missing levels defaulting to a large number to sink them to the end.
4. Define `local_index[order_id] = 0..n_orders-1`.
5. Build `parent_local_idx`:
   - `-1` if root or parent not in `local_index`,
   - else `local_index[parent_id]`.
6. Store `orders_list` and `parent_local_idx` into contiguous flats:
   - `ds_order_offsets[m+1] = ds_order_offsets[m] + n_orders`
   - `order_ids_flat[ord_start:ord_end] = orders_list`
   - `parent_local_idx_flat[ord_start:ord_end] = parent_local_idx`

### Step D — Fast lookups: per-oppset hash tables

We need `order_id → local_order_idx` O(1). We use **open addressing** + **linear probing**.

For each oppset `m`:
- `ht_size = next_pow2_ge(max(4, 2 * n_orders))`  *(~50% load factor)*
- Allocate `keys: int64[ht_size]`, `vals: int32[ht_size]` filled with `-1`.
- For each `(oid, local_idx)` insert:
  - `h = mix64(oid) & (ht_size - 1)`
  - while `vals[h] != -1`: `h = (h + 1) & (ht_size - 1)`
  - `keys[h] = oid; vals[h] = local_idx`
- Append into flats; track `ht_offsets[m+1] = ht_offsets[m] + ht_size`.

### Step E — Map each row to local order index

For each row index `idx ∈ [s,e)`:
- `oid = rows.order_id[idx]`;
- probe the per-oppset hash table to get `local_order_idx`;
- write `row_order_local_idx[idx] = local_order_idx`.

### Outputs (per batch)

#### A) Arrays (NumPy, C-contiguous)

| name                      | shape            | dtype  | meaning                                                |
|---------------------------|------------------|--------|--------------------------------------------------------|
| `ds_row_offsets`          | `(B+1,)`         | int64  | rows slice for `local_oppset_id = m` is `[s:e)`        |
| `ds_order_offsets`        | `(B+1,)`         | int64  | orders slice for `local_oppset_id = m` is `[s:e)`      |
| `order_ids_flat`          | `(Σ n_orders,)`  | int64  | concatenated `orders_list` of all oppsets              |
| `parent_local_idx_flat`   | `(Σ n_orders,)`  | int32  | parent local index per order; `-1` for root            |
| `row_order_local_idx`     | `(N_rows,)`      | int32  | row → local order index lookup                         |
| `ht_offsets`              | `(B+1,)`         | int64  | hash table slab starts per oppset                      |
| `ht_keys_flat`            | `(Σ slots,)`     | int64  | hash table keys (order_id)                             |
| `ht_vals_flat`            | `(Σ slots,)`     | int32  | hash table values (local index, `-1` = empty)          |

**Invariants**
- All arrays above are **aligned** and **C-contiguous**.
- The slots for oppset `m` in each flat array are **contiguous** slices by design.
- `parent_local_idx_flat` uses `-1` to mark roots; there are **no** cross-oppset references.

#### B) DataFrames (Polars)

##### `batch_oppset_index`

| column            | dtype  | description                              |
|-------------------|--------|------------------------------------------|
| `local_oppset_id` | int32  | 0..B-1 within this batch                 |
| `global_oppset_id`| int64  | joined from global index                 |
| `date`            | Date   |                                          |
| `sym`             | Utf8   |                                          |
| `day_i32`         | int32  |                                          |
| `sym_id`          | int32  |                                          |
| `row_start`       | int64  | start row index for this oppset          |
| `row_end`         | int64  | end row index (exclusive)                |
| `n_rows`          | int64  | row_end - row_start                      |
| `ord_start`       | int64  | start order index in `order_ids_flat`    |
| `ord_end`         | int64  | end order index (exclusive)              |
| `n_orders`        | int64  | ord_end - ord_start                      |
| `ht_start`        | int64  | start slot in `ht_*` flats               |
| `ht_size`         | int64  | number of slots in this oppset’s table   |

##### `rows_with_index` (minimal)

| column             | dtype  | description                                         |
|--------------------|--------|-----------------------------------------------------|
| `date`             | Date   |                                                     |
| `sym`              | Utf8   |                                                     |
| `ping_id`          | Struct | untouched unique key                                |
| `order_id`         | Int64  |                                                     |
| `day_i32`          | Int32  |                                                     |
| `sym_id`           | Int32  |                                                     |
| `local_oppset_id`  | Int32  |                                                     |
| `global_oppset_id` | Int64  |                                                     |
| `local_row`        | Int32  | 0..n_rows-1 within oppset                           |
| `batch_row`        | Int64  | 0..N_rows-1 within batch                            |

---

## 6) How to integrate with Numba (to be implemented later)

Below is a **contract** for the later compute module. You can write the kernels without caring where the arrays came from.

### A) Inputs to a Numba kernel (per batch)

- `ds_row_offsets : int64[B+1]`
- `ds_order_offsets : int64[B+1]`
- `row_order_local_idx : int32[N_rows]`
- `parent_local_idx_flat : int32[Σ n_orders]`
- *(Optional)* any additional per-row/per-order arrays you plan to align (features, baseline qty, etc.).

### B) The typical outer loop in Python

```python
# PSEUDOCODE (driver, Python)
for batch in batches:
    bundle = load_index_bundle(batch)               # arrays + frames
    # Optionally: load per-batch features aligned by rows or orders
    for each_epoch in epochs:
        # Pass numpy arrays to a compiled kernel
        run_batch_kernel(
            ds_row_offsets=bundle.ds_row_offsets,
            ds_order_offsets=bundle.ds_order_offsets,
            row_order_local_idx=bundle.row_order_local_idx,
            parent_local_idx_flat=bundle.parent_local_idx_flat,
            # … other aligned arrays …
        )
        # Collect worker outputs keyed by (local_oppset_id, local_order_idx) or by row
        # Persist per-batch outputs
```

### C) The typical inner loop in Numba (concept)

**Do not implement here; this is the shape of the loops.**

```python
# PSEUDOCODE (Numba-njit friendly):
for m in range(B):                                      # iterate oppsets in this batch
    rs = ds_row_offsets[m]; re = ds_row_offsets[m+1]    # row slice for oppset m
    os = ds_order_offsets[m]; oe = ds_order_offsets[m+1]# order slice for oppset m

    # Example: per-order workspace
    # work[os:oe] = 0  # or baseline

    # If your computation is row-driven:
    for r in range(rs, re):
        j = row_order_local_idx[r]      # local order idx in [0..n_orders-1]
        # … update per-order workspace using j …

    # If you need parent aggregation:
    for j in range(os, oe):
        p = parent_local_idx_flat[j]    # local parent within this oppset (or -1)
        # … propagate to p while p != -1 …
```

### D) Emitting results (per batch)

Two common keys:
- **Order-based**: `(local_oppset_id, local_order_idx)` → value array of length `n_orders` per oppset.
- **Row-based**: one value per row (index via `batch_row`).

---

## 7) Reconstitution (after parallel workers finish)

You will get outputs per batch keyed by **local ids**. Convert to **global ids** as follows:

### A) From local oppset id to global oppset id

```python
# Polars / Python
g = bundle.batch_oppset_index.filter(pl.col("local_oppset_id") == m)["global_oppset_id"][0]
# or pre-extract a numpy array indexed by local_oppset_id for speed.
```

### B) From local order idx to order_id

```python
os = bundle.ds_order_offsets[m]; oe = bundle.ds_order_offsets[m+1]
order_id = bundle.order_ids_flat[os + local_order_idx]
```

### C) Now you have `(global_oppset_id, order_id, value)`

Join with the **GlobalOppsetIndex** (to get `(date, sym)`), or merge into your original row-level tables via `(date, sym)` or `(date, order_id)` as needed.

**Note**: If you emitted row-based values, use `batch_row` to align to the batch’s oppset rows, and optionally `(date, sym, ping_id)` to merge with analytics tables.

---

## 8) I/O Recommendations (lightweight)

- **Global index**: Parquet (`global_oppset_index.parquet`).
- **Per batch**:
  - DataFrames: Parquet (`batch_oppset_index.parquet`, `rows_with_index.parquet`).
  - Arrays: a single `.npz` (compressed) holding all flats:  
    `ds_row_offsets, ds_order_offsets, order_ids_flat, parent_local_idx_flat, row_order_local_idx, ht_offsets, ht_keys_flat, ht_vals_flat`.

**Naming**:
```
/indexes/
  global/
    global_oppset_index.parquet
  batches/
    batch_0001/
      batch_oppset_index.parquet
      rows_with_index.parquet
      index_arrays.npz
    batch_0002/
      ...
```

---

## 9) Validation & Edge Cases

- **Missing parent** in orders: ancestor walk stops if `parent_id` missing or `< 0` or `parent == id`.
- **Cycles**: if ancestor walk exceeds a safety threshold (e.g., 1e6 steps), raise.
- **Empty oppset** (no rows): allowed; yields 0-length order slice; hash table size = 1 with `-1` sentinel.
- **Deterministic order** within oppset: sorting by `(level asc, order_id asc)` stabilizes local order indices.
- **Type stability**: cast exactly as specified (`int32`/`int64`).

---

## 10) Performance Notes

- The per-day Python dicts (`parent_by_day`, `level_by_day`) are key for fast ancestor closure.
- Hash tables use **~50% load factor** by sizing to `pow2(2 * n_orders)`; probing is linear and branch-light.
- All flats are contiguous; workers can slice without allocation.
- For massive datasets, write arrays with **`np.savez_compressed`** and `mmap_mode="r"` on load if desired.
- For Polars, keep grouping keys pre-cast to avoid repeated casts; push filtering early.

---

## 11) Minimal API Surface (what another agent needs)

### Build once (driver)

```python
global_idx = build_global_oppset_index(all_oppsets_df)
global_idx.write_parquet("/indexes/global/global_oppset_index.parquet")
```

### Per batch

```python
bundle = build_batch_local_indices(batch_oppsets_df, batch_orders_df, global_idx)

# persist
bundle.batch_oppset_index.write_parquet(".../batch_oppset_index.parquet")
bundle.rows_with_index.write_parquet(".../rows_with_index.parquet")
np.savez_compressed(".../index_arrays.npz",
    ds_row_offsets=bundle.ds_row_offsets,
    ds_order_offsets=bundle.ds_order_offsets,
    order_ids_flat=bundle.order_ids_flat,
    parent_local_idx_flat=bundle.parent_local_idx_flat,
    row_order_local_idx=bundle.row_order_local_idx,
    ht_offsets=bundle.ht_offsets,
    ht_keys_flat=bundle.ht_keys_flat,
    ht_vals_flat=bundle.ht_vals_flat
)
```

### Worker (load + compute)

```python
# load parquet + npz (arrays into numpy, contiguous)
# hand arrays to Numba kernels that follow the loop contracts above
```

### Driver (reassemble)

```python
# For each batch:
#   1) Load its batch_oppset_index + arrays
#   2) Convert worker keys (local_oppset_id, local_order_idx) → (global_oppset_id, order_id)
#   3) Merge across batches; join to global index and/or original tables
```

---

## 12) Test Recipe (should pass before wiring compute)

1. **Index determinism**: run builders twice on the same inputs; compare `global_oppset_id`, `local_oppset_id`, `ds_*_offsets`.
2. **Ancestor closure**: for a sample oppset, ensure `orders_list` includes all expected roots/parents.
3. **Hash table probes**: for each order in `orders_list`, `probe(order_id)` returns its local index.
4. **Row mapping**: for each row `r` in oppset `m`, check `orders_list[row_order_local_idx[r]] == row.order_id`.
5. **Reconstitution**: fabricate outputs per oppset with a known pattern (e.g., `value = local_order_idx`), reassemble to `(global_oppset_id, order_id)`, verify join cardinality and exact mapping.

---

## 13) Future Extensions (non-breaking)

- Add an optional **order sort policy** (e.g., by a user-specified stable key) instead of `(level, order_id)`.
- Carry **feature names** (or IDs) per row/order alongside indices, but keep the compute path array-only.
- Provide a compact **C ABI** for kernels (struct-of-arrays header with pointers/lengths).

---

## 14) Glossary

- **Oppset**: All rows for a single `(date, sym)` (a mini dataset).  
- **Batch**: A collection of oppsets processed together (usually for parallelism).  
- **Ancestor closure**: The union of orders in the oppset plus all their parents up to the root.  
- **Local order index**: A compact `0..n_orders-1` numbering within an oppset.  
- **Hash slab**: The contiguous region in `ht_*` arrays used by a single oppset.

---

## 15) Checklist for the Agent Implementing Numba Kernels

- [ ] Accept arrays with the exact shapes/dtypes above.  
- [ ] Iterate oppsets via `ds_row_offsets` and `ds_order_offsets`.  
- [ ] Use `row_order_local_idx` to map row-driven logic to per-order state.  
- [ ] Use `parent_local_idx_flat` for upward-only propagation (if/when added).  
- [ ] Emit results keyed by `(local_oppset_id, local_order_idx)` or by `batch_row`.  
- [ ] Do **not** assume `order_id` continuity; rely on local indices only.  
- [ ] Avoid Python objects in Numba (no dicts); all inputs must be NumPy arrays.  
- [ ] Keep arrays C-contiguous; pre-allocate outputs; avoid per-iteration allocs.  
- [ ] Document the kernel’s assumptions and invariants; add assertions behind a debug flag.

---

**That’s the whole contract.** With this, another agent can implement the Polars→NumPy→Numba compute module and wire inner/outer loops without guessing. The existing `oppset_batch_indexing.py` already produces everything specified here.
