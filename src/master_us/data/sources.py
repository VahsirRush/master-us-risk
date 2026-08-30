"""Free-path data acquisition — the module contract of data-sources-contract section 1.

Every implementation here is cache-first: check parquet, fetch only on miss,
write immediately. Raw pulls under `data/raw/` are immutable — never mutate
anything there, only append.

The three Protocols exist so that a later WRDS or Sharadar upgrade is a drop-in
swap rather than a rewrite. Nothing downstream of `Panel` should ever import a
concrete source class.

This module holds the Protocols and the infrastructure every source shares:
the SEC user-agent guard, the rate limiter, and the retry wrapper. Concrete
implementations live next door in `universe.py`, `loaders.py`, and `sec.py`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar

import polars as pl
import yaml

T = TypeVar("T")

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
CONFIG_ROOT = REPO_ROOT / "config"


# --------------------------------------------------------------------- #
# Protocols — data-sources-contract section 1                            #
# --------------------------------------------------------------------- #


class PriceSource(Protocol):
    def get_ohlcv(self, tickers: list[str], start: str, end: str) -> pl.DataFrame:
        """Long format: [date, ticker, open, high, low, close, adj_close, volume].

        Split- and dividend-adjusted. Missing days are ABSENT, never NaN-filled —
        a NaN row would be indistinguishable from a genuine trading halt.
        """
        ...


class FundamentalSource(Protocol):
    def get_fundamentals(self, start: str, end: str) -> pl.DataFrame:
        """Long format: [ticker, cik, filed, period_end, tag, value].

        `filed` is the SEC acceptance date and is the ONLY date usable for PIT
        joins. `period_end` is present for diagnostics and must never reach a
        join key.
        """
        ...


class UniverseSource(Protocol):
    def get_membership(self, start: str, end: str) -> pl.DataFrame:
        """Long format: [date, ticker, in_universe: bool].

        Daily, forward-filled from monthly reconstitution.
        """
        ...


# --------------------------------------------------------------------- #
# The SEC user-agent guard — data-sources-contract section 4.4           #
# --------------------------------------------------------------------- #

# Substrings that mean the config was copied but never filled in. The SEC
# blocks by IP for anonymous or spoofed agents, and a block is slow to lift, so
# this fails before the first request rather than after it.
PLACEHOLDER_MARKERS: tuple[str, ...] = (
    "REPLACE-ME",
    "REPLACE_ME",
    "your-email@domain.com",
    "your.email@",
    "example.com",
    "example.org",
    "user@domain",
    "TODO",
)


class InvalidUserAgentError(RuntimeError):
    """Raised when the configured SEC User-Agent is a placeholder or malformed."""


def require_valid_user_agent(user_agent: str | None) -> str:
    """Validate the SEC User-Agent, or refuse to let the caller reach the endpoint.

    The SEC requires a declared agent carrying a real name and email. Sending a
    placeholder is worse than sending nothing: it is the pattern their abuse
    detection looks for, and the resulting IP block affects everything on the
    connection, not just this project.

    Deliberately defensive. The config is filled in today, but configs get
    copied to new machines, reset by a merge, or templated for someone else —
    and the failure mode without this check is an IP block rather than an error.
    """
    if user_agent is None or not user_agent.strip():
        raise InvalidUserAgentError(
            "SEC User-Agent is empty. Set fundamentals.user_agent in config/data.yaml "
            "to 'Your Name <you@domain.com>' before fetching from sec.gov."
        )

    agent = user_agent.strip()
    hit = next((m for m in PLACEHOLDER_MARKERS if m.lower() in agent.lower()), None)
    if hit is not None:
        raise InvalidUserAgentError(
            f"SEC User-Agent still contains the placeholder {hit!r}: {agent!r}. "
            "The SEC blocks by IP for placeholder or absent agents, and the block is "
            "slow to lift. Set fundamentals.user_agent in config/data.yaml to a real "
            "name and email before fetching from sec.gov."
        )

    if "@" not in agent or "." not in agent.split("@", 1)[1]:
        raise InvalidUserAgentError(
            f"SEC User-Agent must contain a contactable email address, got {agent!r}. "
            "Expected the form 'Your Name <you@domain.com>'."
        )
    return agent


# --------------------------------------------------------------------- #
# Shared infrastructure                                                  #
# --------------------------------------------------------------------- #


@dataclass
class RateLimiter:
    """Blocking token bucket, deliberately set under the published cap.

    The SEC caps at roughly 10 requests/second. `config/data.yaml` asks for 8.
    """

    per_second: float
    _last: float = field(default=0.0, repr=False)

    def wait(self) -> None:
        if self.per_second <= 0:
            return
        gap = 1.0 / self.per_second
        elapsed = time.monotonic() - self._last
        if elapsed < gap:
            time.sleep(gap - elapsed)
        self._last = time.monotonic()


class FetchError(RuntimeError):
    """A fetch that failed every attempt. Carries the per-attempt reasons."""

    def __init__(self, what: str, attempts: Sequence[str]) -> None:
        self.what = what
        self.attempts = list(attempts)
        detail = "; ".join(f"[{i + 1}] {a}" for i, a in enumerate(attempts))
        super().__init__(f"{what} failed after {len(attempts)} attempts: {detail}")


def with_retry(
    fn: Callable[[], T],
    what: str,
    backoff: Sequence[float] = (2.0, 8.0, 32.0),
    on_attempt: Callable[[int, Exception], None] | None = None,
) -> T:
    """Run `fn`, retrying with explicit backoff. Raises `FetchError` if all fail.

    Never returns a sentinel on failure. CLAUDE.md rule 6: a caller that gets a
    value back must be able to trust it, and a silent empty frame from a
    throttled endpoint is exactly the degradation this project must not have.
    """
    reasons: list[str] = []
    for attempt, delay in enumerate([0.0, *backoff]):
        if delay:
            time.sleep(delay)
        try:
            return fn()
        except Exception as exc:  # re-raised as FetchError below; never swallowed
            reasons.append(f"{type(exc).__name__}: {exc}")
            if on_attempt is not None:
                on_attempt(attempt, exc)
    raise FetchError(what, reasons)


def batched(items: Sequence[T], size: int) -> Iterator[list[T]]:
    """Yield consecutive chunks of `size`."""
    if size < 1:
        raise ValueError(f"batch size must be >= 1, got {size}")
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def load_data_config(path: Path | None = None) -> dict[str, Any]:
    """Read `config/data.yaml`."""
    cfg_path = path or (CONFIG_ROOT / "data.yaml")
    with cfg_path.open() as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise TypeError(f"{cfg_path} did not parse to a mapping")
    return loaded


def sec_user_agent(cfg: dict[str, Any] | None = None) -> str:
    """The validated SEC User-Agent from config. Raises rather than returning junk."""
    config = cfg if cfg is not None else load_data_config()
    return require_valid_user_agent(config.get("fundamentals", {}).get("user_agent"))
