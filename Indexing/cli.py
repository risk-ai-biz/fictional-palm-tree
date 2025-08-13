from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from .oppset_batch_indexing import (
    build_global_oppset_index,
    build_batch_local_indices,
    save_global_index,
    save_batch_bundle,
    load_global_index,
    load_batch_bundle,
)


def _cmd_build_global(args: argparse.Namespace) -> None:
    oppsets = pl.read_parquet(args.oppsets)
    global_idx = build_global_oppset_index(oppsets)
    out_dir = Path(args.out_dir) / "global"
    save_global_index(global_idx, out_dir, overwrite=args.overwrite)


def _cmd_build_batch(args: argparse.Namespace) -> None:
    global_idx = load_global_index(args.global_index)
    oppsets = pl.read_parquet(args.oppsets)
    orders = pl.read_parquet(args.orders)
    bundle = build_batch_local_indices(oppsets, orders, global_idx)
    batch_dir = Path(args.out_dir) / "batches" / f"batch_{args.batch_id:04d}"
    save_batch_bundle(bundle, batch_dir, overwrite=args.overwrite)


def _cmd_load_batch(args: argparse.Namespace) -> None:
    bundle = load_batch_bundle(args.batch_dir, mmap=args.mmap)
    print("batch_oppset_index rows:", bundle.batch_oppset_index.height)
    print("rows_with_index rows:", bundle.rows_with_index.height)
    print("ds_row_offsets shape:", bundle.ds_row_offsets.shape)
    print("ds_order_offsets shape:", bundle.ds_order_offsets.shape)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Oppset indexing helpers")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_global = sub.add_parser("build-global", help="Build the global oppset index")
    p_global.add_argument("oppsets", help="Parquet file with all oppsets")
    p_global.add_argument("out_dir", help="Output root directory")
    p_global.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p_global.set_defaults(func=_cmd_build_global)

    p_batch = sub.add_parser("build-batch", help="Build indices for a dataset batch")
    p_batch.add_argument("oppsets", help="Parquet file with batch oppsets")
    p_batch.add_argument("orders", help="Parquet file with batch orders")
    p_batch.add_argument("global_index", help="Path to global_oppset_index.parquet or its directory")
    p_batch.add_argument("out_dir", help="Output root directory")
    p_batch.add_argument("batch_id", type=int, help="Batch number (1-based)")
    p_batch.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p_batch.set_defaults(func=_cmd_build_batch)

    p_load = sub.add_parser("load-batch", help="Load a saved batch bundle and print summary")
    p_load.add_argument("batch_dir", help="Directory containing batch bundle files")
    p_load.add_argument("--mmap", action="store_true", help="Memory-map arrays read-only")
    p_load.set_defaults(func=_cmd_load_batch)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
