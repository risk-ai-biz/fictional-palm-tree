"""Command line helpers for working with *oppset* indices.

This module exposes a small ``argparse`` driven CLI with a handful of
subcommands.  It can build the global index, construct per-batch indices and
inspect a previously generated batch bundle.  The code is intentionally
light‑weight so it can be imported and reused in other tooling as well as run
directly from the terminal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from .oppset_batch_indexing import (
    build_batch_local_indices,
    build_global_oppset_index,
    load_batch_bundle,
    load_global_index,
    save_batch_bundle,
    save_global_index,
)

def _cmd_build_global(args: argparse.Namespace) -> None:
    """Build the global oppset index and write it to disk."""

    # Load every oppset row from the provided Parquet file.
    oppsets = pl.read_parquet(args.oppsets)

    # Construct a single index that covers the entire dataset.
    global_idx = build_global_oppset_index(oppsets)

    # Persist the index inside ``<out_dir>/global`` so that it can be reused
    # when building per-batch indices.
    out_dir = Path(args.out_dir) / "global"
    save_global_index(global_idx, out_dir, overwrite=args.overwrite)

def _cmd_build_batch(args: argparse.Namespace) -> None:
    """Build indices for a single batch of data."""

    # The batch-specific indices reference rows in the global index, so load it
    # first.
    global_idx = load_global_index(args.global_index)

    # Read batch-level oppset information and the corresponding orders.
    oppsets = pl.read_parquet(args.oppsets)
    orders = pl.read_parquet(args.orders)

    # Construct the set of batch-local indices.
    bundle = build_batch_local_indices(oppsets, orders, global_idx)

    # Store the results under ``<out_dir>/batches/batch_<id>`` for later use.
    batch_dir = Path(args.out_dir) / "batches" / f"batch_{args.batch_id:04d}"
    save_batch_bundle(bundle, batch_dir, overwrite=args.overwrite)

def _cmd_load_batch(args: argparse.Namespace) -> None:
    """Load a batch bundle and print a small summary for inspection."""

    # This command is mostly for debugging or quick exploration, so we simply
    # report basic statistics about the stored arrays.
    bundle = load_batch_bundle(args.batch_dir, mmap=args.mmap)
    print("batch_oppset_index rows:", bundle.batch_oppset_index.height)
    print("rows_with_index rows:", bundle.rows_with_index.height)
    print("ds_row_offsets shape:", bundle.ds_row_offsets.shape)
    print("ds_order_offsets shape:", bundle.ds_order_offsets.shape)

def build_parser() -> argparse.ArgumentParser:
    """Create the top level argument parser for the CLI."""

    parser = argparse.ArgumentParser(description="Oppset indexing helpers")

    # All functionality lives behind subcommands: ``build-global`` generates
    # the global index, ``build-batch`` produces per-batch files and
    # ``load-batch`` prints a quick summary of a saved bundle.
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- build-global -----------------------------------------------------
    p_global = sub.add_parser("build-global", help="Build the global oppset index")
    p_global.add_argument("oppsets", help="Parquet file with all oppsets")
    p_global.add_argument("out_dir", help="Output root directory")
    p_global.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p_global.set_defaults(func=_cmd_build_global)

    # --- build-batch ------------------------------------------------------
    p_batch = sub.add_parser("build-batch", help="Build indices for a dataset batch")
    p_batch.add_argument("oppsets", help="Parquet file with batch oppsets")
    p_batch.add_argument("orders", help="Parquet file with batch orders")
    p_batch.add_argument(
        "global_index", help="Path to global_oppset_index.parquet or its directory"
    )
    p_batch.add_argument("out_dir", help="Output root directory")
    p_batch.add_argument("batch_id", type=int, help="Batch number (1-based)")
    p_batch.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p_batch.set_defaults(func=_cmd_build_batch)

    # --- load-batch -------------------------------------------------------
    p_load = sub.add_parser("load-batch", help="Load a saved batch bundle and print summary")
    p_load.add_argument("batch_dir", help="Directory containing batch bundle files")
    p_load.add_argument("--mmap", action="store_true", help="Memory-map arrays read-only")
    p_load.set_defaults(func=_cmd_load_batch)

    return parser

def main(argv: list[str] | None = None) -> None:
    """Entry point used by ``python -m`` or console scripts."""

    parser = build_parser()
    args = parser.parse_args(argv)

    # Each subcommand stores a callback in ``func`` that performs the actual
    # work, so dispatch to it here.
    args.func(args)


if __name__ == "__main__":
    main()
