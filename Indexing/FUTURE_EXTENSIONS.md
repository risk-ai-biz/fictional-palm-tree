# Future Extensions

This document sketches forward-looking options for evolving the order/indexing
layer without breaking existing contracts.

## 1) Pluggable order-sort policies

**Design options**

- **Policy hook at build time** – allow `build_batch_local_indices` to accept a
  callable that returns a sort key per order. Builders apply the key after
  ancestor expansion to produce the `order_ids_flat` and related arrays.
- **Declarative spec** – accept a list of column names and sort directions
  resolved from the input tables. The builder constructs the composite key and
  falls back to `(level, order_id)` when unspecified.

**Implications**

- Builders must expose the new hook while preserving current behaviour when the
  policy is `None`.
- Kernels relying on `(level, order_id)` ordering must either remain compatible
  or surface an explicit policy requirement.

**Open questions**

- Should policies be restricted to pure Python callables, or support Polars
  expressions for zero-copy sorting?
- How are stability and determinism guaranteed across processes and languages?
- Do kernels need to report the policy they assume for validation?

## 2) Feature metadata alongside indices

**Design options**

- **Parallel arrays** – store feature IDs or names in arrays aligned with
  `row_order_local_idx`. Kernels receive only numeric arrays; higher-level code
  keeps the metadata for interpretation.
- **External dictionary** – persist a mapping of `(feature_id → string)` per
  batch. Builders attach feature IDs to rows; kernels operate purely on IDs.

**Implications**

- Builders gain optional paths for emitting metadata files alongside existing
  `npz` bundles.
- Kernels remain unchanged as long as they treat feature IDs as opaque integers.
- Downstream consumers can join on feature metadata without modifying kernel
  logic.

**Open questions**

- How are feature IDs allocated to avoid collisions across batches?
- Is metadata static, or can kernels request subsets at load time?
- What is the storage format for large numbers of features (e.g., Parquet vs
  JSON)?

## 3) Minimal C ABI for kernel interoperability

**Design options**

- **Struct-of-arrays header** – define a C struct containing pointers and
  lengths for each required array (`ds_row_offsets`, `order_ids_flat`, etc.).
  Language bindings populate the struct and pass it to kernels.
- **Flat pointer table** – expose arrays as `void**` with a parallel enum that
  identifies each slot. This keeps the ABI stable even if fields are appended.

**Implications**

- Builders must ensure arrays are contiguous and exportable via `ctypes` or
  similar mechanisms.
- Existing Python/NumPy kernels can adopt the ABI gradually through wrapper
  functions.
- Cross-language kernels (C/C++/Rust) gain a stable entry point decoupled from
  Python object models.

**Open questions**

- How is memory ownership managed across the boundary (caller vs callee)?
- What alignment guarantees are required for SIMD-heavy kernels?
- Should the ABI cover error reporting and logging, or remain computation-only?

