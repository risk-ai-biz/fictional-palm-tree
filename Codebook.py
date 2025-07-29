SPDX-License-Identifier: MIT

"""codebook.py – minimal, pandas‑free categorical encoder / decoder

Motivation

When you push data into Numba you want **dense int32 codes** and a flat
`np.ndarray` lookup – no hashes, no Python objects.  This helper turns
one or many string / categorical columns in a **Polars** DataFrame into
exactly that *once* at ingest time, then stays out of the hot path.

Key points

Polars → NumPy only – zero pandas dependency.

Column life‑cycle verbs are build → encode → decode (no “fit”).

Bidirectional mapping kept alongside the lookup arrays so reversing is always loss‑free.

JSON serialisable metadata for reproducible back‑tests.


If you do hand the frame to pandas downstream, just call polars_df.to_pandas() after encoding. """ from future import annotations

import json import pathlib from dataclasses import dataclass from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np import polars as pl

all = ["CodeBook", "CodeBookError"]

class CodeBookError(RuntimeError): """Raised for any misuse of :class:CodeBook."""

---------------------------------------------------------------------

Internal helpers

---------------------------------------------------------------------

@dataclass(slots=True) class _ColMeta: labels: np.ndarray      # 1‑D lookup array, dtype preserved null_code: Optional[int]

def as_dict(self) -> Dict[str, Any]:
    # numpy -> list so JSON serialises cleanly
    return {
        "labels": self.labels.tolist(),
        "null_code": self.null_code,
    }

@classmethod
def from_dict(cls, d: Mapping[str, Any]) -> "_ColMeta":
    return cls(labels=np.asarray(d["labels"]), null_code=d["null_code"])

---------------------------------------------------------------------

Public API

---------------------------------------------------------------------

class CodeBook: """Manage loss‑less encoding ⇄ decoding of categorical columns.

Parameters
----------
null_bucket : bool, default True
    If *True* adds one extra label so that *every* value (incl. NA)
    maps to a **non‑negative** code.
suffix : str, default "__c"
    Name postfix for the encoded int32 column.
sort_labels : bool, default True
    Sort labels lexicographically before assigning codes so builds
    are deterministic.
"""

def __init__(self, *, null_bucket: bool = True, suffix: str = "__c", sort_labels: bool = True):
    self._null_bucket: bool = null_bucket
    self._suffix: str = suffix
    self._sort: bool = sort_labels
    self._meta: Dict[str, _ColMeta] = {}

# .................................................................
# BUILD – discover dictionaries
# .................................................................
def build(self, df: pl.DataFrame, cols: Sequence[str]) -> None:
    """Remember unique labels for *cols* in *df*.

    Can be called **only once per column**; re‑building overwrites.
    """
    for col in cols:
        if col not in df.columns:
            raise CodeBookError(f"'{col}' not in DataFrame")
        # Polars unique is already hash‑based & fast
        labels = df.get_column(col).drop_nans().unique().to_numpy()
        if self._sort:
            labels = np.sort(labels)

        null_code: Optional[int] = None
        if self._null_bucket:
            # Pick a null label of the same dtype
            if labels.dtype.kind in {"U", "S"}:  # unicode / bytes
                null_label = ""  # empty str
            else:
                null_label = np.nan
            labels = np.concatenate([labels, np.asarray([null_label], dtype=labels.dtype)])
            null_code = labels.size - 1

        self._meta[col] = _ColMeta(labels=labels, null_code=null_code)

# .................................................................
# ENCODE – Polars -> Polars with int32 codes
# .................................................................
def encode(self, df: pl.DataFrame, *, cols: Optional[Sequence[str]] = None) -> pl.DataFrame:
    if cols is None:
        cols = tuple(self._meta.keys())
    unknown = set(cols) - self._meta.keys()
    if unknown:
        raise CodeBookError(f"columns not in codebook: {unknown}")

    out = df.clone()
    for col in cols:
        meta = self._meta[col]
        code_col = f"{col}{self._suffix}"
        label2code = {lab: i for i, lab in enumerate(meta.labels)}
        default_code = meta.null_code if meta.null_code is not None else -1
        # Polars expression replaces per element via hash lookup – vectorised in Rust.
        out = out.with_columns(
            pl.col(col)
            .map_dict(label2code, default=default_code)
            .cast(pl.Int32)
            .alias(code_col)
        )
    return out

# .................................................................
# DECODE – int codes back to original dtype
# .................................................................
def decode(self, df: pl.DataFrame, *, cols: Optional[Sequence[str]] = None) -> pl.DataFrame:
    if cols is None:
        cols = tuple(self._meta.keys())
    unknown = set(cols) - self._meta.keys()
    if unknown:
        raise CodeBookError(f"columns not in codebook: {unknown}")

    out = df.clone()
    for col in cols:
        meta = self._meta[col]
        code_col = f"{col}{self._suffix}"
        if code_col not in out.columns:
            raise CodeBookError(f"expected '{code_col}' in DataFrame")
        rev_map = {int(i): lab for i, lab in enumerate(meta.labels)}
        out = out.with_columns(
            pl.col(code_col)
            .map_dict(rev_map, default=None)  # returns original dtype
            .alias(col)
        )
    return out

# .................................................................
# RUNTIME LOOKUPS – hand to Numba
# .................................................................
def lookup_array(self, col: str) -> np.ndarray:
    try:
        return self._meta[col].labels
    except KeyError as err:
        raise CodeBookError(f"unknown column '{col}'") from err

def code_for_null(self, col: str) -> Optional[int]:
    return self._meta[col].null_code

# .................................................................
# SERIALISATION – JSON round‑trip
# .................................................................
def to_dict(self) -> Dict[str, Any]:
    return {name: meta.as_dict() for name, meta in self._meta.items()}

def save(self, path: str | pathlib.Path) -> None:
    pathlib.Path(path).write_text(json.dumps(self.to_dict(), separators=(",", ":")))

@classmethod
def load(cls, path: str | pathlib.Path, *, suffix: str = "__c") -> "CodeBook":
    raw = pathlib.Path(path).read_text()
    data = json.loads(raw)
    self = cls(suffix=suffix)
    self._meta = {n: _ColMeta.from_dict(m) for n, m in data.items()}
    return self

# .................................................................
# Debug helpers
# .................................................................
def __repr__(self) -> str:  # pragma: no cover
    return (
        f"CodeBook({len(self._meta)} cols) "
        + ", ".join(f"{k}:{v.labels.size}" for k, v in self._meta.items())
    )

