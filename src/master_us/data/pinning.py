"""Price-data content pinning — data-sources-contract section 3.2.

Session 4 caught Yahoo re-basing LEG's adjustments between two pulls by luck:
the monotonicity check happened to straddle the merge seam. This module makes
the detection structural. After a pull passes validation, `pin` stores a
content hash per ticker; every subsequent consumer calls `verify` before
trusting the data, and a mismatch raises naming the tickers that changed.

Hashes cover the DATA, not the file bytes: per ticker, a SHA-256 over the
canonical little-endian byte layout of (date, open, high, low, close,
adj_close, volume) sorted by date. Re-writing the same values with a
different parquet compressor or library version does not trip the pin;
changing one adjusted close by one ULP does.

Policy on the three kinds of drift:

* CHANGED ticker — error, always. That is the LEG failure mode: history
  silently rewritten under the same name.
* MISSING ticker (pinned but gone) — error. Deleting raw data is not a thing
  that happens by accident.
* NEW ticker (present but unpinned) — reported, not an error. The cache is
  append-only by design; extending it with new names is legitimate growth.
  Re-run the pin to adopt them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from master_us.data.sources import RAW_ROOT

PRICE_MANIFEST_PATH = RAW_ROOT / "price_manifest.json"

HASHED_COLUMNS = ("open", "high", "low", "close", "adj_close", "volume")


class PriceDataChangedError(RuntimeError):
    """Pinned price history no longer matches what is on disk."""


@dataclass(frozen=True)
class VerificationResult:
    """What `verify` found. `ok` means nothing pinned has drifted."""

    n_checked: int
    changed: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    new: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.changed and not self.missing


def _hash_frame(frame: pl.DataFrame) -> str:
    """Canonical content hash of one ticker's history."""
    ordered = frame.sort("date")
    hasher = hashlib.sha256()
    hasher.update(ordered["date"].cast(pl.Int64).to_numpy().astype("<i8").tobytes())
    for col in HASHED_COLUMNS:
        hasher.update(np.ascontiguousarray(ordered[col].to_numpy(), dtype="<f8").tobytes())
    return hasher.hexdigest()


def _hash_dir(price_dir: Path) -> dict[str, str]:
    return {
        path.stem: _hash_frame(pl.read_parquet(path))
        for path in sorted(price_dir.glob("*.parquet"))
    }


def pin(price_dir: Path, manifest_path: Path = PRICE_MANIFEST_PATH) -> dict[str, str]:
    """Hash every cached ticker and write the manifest. Returns the hashes."""
    hashes = _hash_dir(price_dir)
    if not hashes:
        raise FileNotFoundError(f"nothing to pin under {price_dir}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as fh:
        json.dump(
            {
                "pinned_at": datetime.now(UTC).isoformat(),
                "price_dir": str(price_dir),
                "n_tickers": len(hashes),
                "hashes": hashes,
            },
            fh,
            indent=2,
            sort_keys=True,
        )
    return hashes


def verify(
    price_dir: Path,
    manifest_path: Path = PRICE_MANIFEST_PATH,
    raise_on_drift: bool = True,
) -> VerificationResult:
    """Recompute hashes and compare against the manifest.

    Raises `PriceDataChangedError` naming the drifted tickers unless
    `raise_on_drift=False` (used by reporting paths that want the list).
    A missing manifest also raises: consuming unpinned data is the state
    this module exists to end.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} does not exist — run scripts/08_pin_prices.py after a "
            "validated pull. Consuming unpinned price data is how the LEG mixed-basis "
            "bug happened."
        )
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    pinned: dict[str, str] = manifest["hashes"]

    current = _hash_dir(price_dir)
    changed = tuple(sorted(t for t in pinned if t in current and current[t] != pinned[t]))
    missing = tuple(sorted(t for t in pinned if t not in current))
    new = tuple(sorted(t for t in current if t not in pinned))

    result = VerificationResult(
        n_checked=len(pinned), changed=changed, missing=missing, new=new
    )
    if not result.ok and raise_on_drift:
        parts = []
        if changed:
            parts.append(
                f"{len(changed)} ticker(s) have DIFFERENT history than when pinned: "
                f"{list(changed[:10])} — Yahoo has re-based adjustments (the LEG "
                "failure mode). Refetch the affected tickers' FULL history in one "
                "pull, re-validate, then re-pin deliberately."
            )
        if missing:
            parts.append(f"{len(missing)} pinned ticker(s) are gone from disk: {list(missing[:10])}")
        raise PriceDataChangedError(
            f"price data drifted from the manifest pinned at {manifest['pinned_at']}: "
            + " ".join(parts)
        )
    return result
