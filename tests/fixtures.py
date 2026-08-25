"""Synthetic Panel construction.

Two purposes:

1. Let every downstream module be built and tested before any real data
   exists. Phases 1 through 8 can all be exercised against synthetic panels.

2. Provide *corrupted* panels for the leakage tests. A test that only ever
   passes is not a test — each Phase-0 test must be shown to fail on a panel
   with the corresponding defect deliberately introduced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from master_us.data.panel import Panel


def make_synthetic_panel(
    n_dates: int = 500,
    n_tickers: int = 50,
    n_features: int = 20,
    n_market: int = 8,
    seed: int = 0,
    signal_strength: float = 0.15,
    signal_feature: int = 0,
    missing_rate: float = 0.05,
    label_horizon: int = 1,
    with_metadata: bool = True,
    n_delistings: int = 0,
) -> Panel:
    """A valid Panel with a known injected cross-sectional signal.

    `features[:, :, signal_feature]` predicts the label with correlation
    approximately `signal_strength`. Any model worth its name must recover it,
    and any leakage test must not find MORE than it.

    `n_delistings > 0` gives that many tickers a delisting date partway through
    the sample: in the universe up to and including that date, with a terminal
    return on it, and absent thereafter. Defaults to 0 so that every panel built
    before this option existed is bit-identical — the extra RNG draws happen
    only when delistings are requested.
    """
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range("2015-01-01", periods=n_dates, freq="C")
    tickers = np.array([f"T{i:04d}" for i in range(n_tickers)])

    features = rng.standard_normal((n_dates, n_tickers, n_features)).astype(np.float32)
    market = rng.standard_normal((n_dates, n_market)).astype(np.float32)

    # Label is the signal feature plus noise, cross-sectionally demeaned.
    noise = rng.standard_normal((n_dates, n_tickers)).astype(np.float32)
    raw = signal_strength * features[:, :, signal_feature] + np.sqrt(
        max(1.0 - signal_strength**2, 0.0)
    ) * noise
    labels = (raw - raw.mean(axis=1, keepdims=True)).astype(np.float32)

    mask = rng.random((n_dates, n_tickers)) > missing_rate
    mask[:, 0] = True  # guarantee no empty date
    features[~mask] = np.nan
    labels[~mask] = np.nan

    delisted: dict[str, int] = {}
    if n_delistings > 0:
        delisted = _apply_delistings(
            features, labels, mask, tickers, n_delistings, label_horizon, rng
        )

    # Final `label_horizon` rows have no realized forward return.
    labels[-label_horizon:, :] = np.nan

    metadata = _make_metadata(dates, tickers, mask, rng) if with_metadata else None

    return Panel(
        dates=dates,
        tickers=tickers,
        features=features,
        feature_names=[f"f{i:03d}" for i in range(n_features)],
        market=market,
        market_names=[f"m{i:02d}" for i in range(n_market)],
        labels=labels,
        mask=mask,
        label_horizon=label_horizon,
        metadata=metadata,
        attrs={
            "synthetic": True,
            "seed": seed,
            "signal_feature": signal_feature,
            "delisted": delisted,
        },
    )


def _apply_delistings(
    features: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    tickers: np.ndarray,
    n_delistings: int,
    label_horizon: int,
    rng: np.random.Generator,
) -> dict[str, int]:
    """Give `n_delistings` tickers a terminal date, mutating the arrays in place.

    A delisted name is tradeable through its delisting date, carries a realized
    terminal return ON that date, and is absent afterwards. This is what a panel
    built correctly looks like; `corrupt_survivorship_drop_delisted` is what one
    built with survivorship bias looks like.

    Ticker 0 is never chosen — it is the pin that keeps every date non-empty.
    """
    n_dates, n_tickers = mask.shape
    lo = max(int(n_dates * 0.2), 1)
    hi = max(n_dates - label_horizon - 1, lo + 1)
    if n_delistings >= n_tickers:
        raise ValueError(f"n_delistings={n_delistings} must be < n_tickers={n_tickers}")

    doomed = rng.choice(np.arange(1, n_tickers), size=n_delistings, replace=False)
    out: dict[str, int] = {}
    for col in doomed:
        d = int(rng.integers(lo, hi))
        mask[d + 1 :, col] = False
        features[d + 1 :, col, :] = np.nan
        labels[d + 1 :, col] = np.nan

        # Tradeable on the delisting date, with the terminal return applied.
        mask[d, col] = True
        nan_feats = np.isnan(features[d, col, :])
        if nan_feats.any():
            features[d, col, nan_feats] = rng.standard_normal(int(nan_feats.sum())).astype(
                np.float32
            )
        labels[d, col] = np.float32(rng.normal(-0.35, 0.20))
        out[str(tickers[col])] = d
    return out


def _make_metadata(
    dates: pd.DatetimeIndex,
    tickers: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    n = len(idx)
    sectors = rng.choice(
        ["Tech", "Financials", "Health", "Energy", "Industrials"], size=len(tickers)
    )
    return pd.DataFrame(
        {
            "mcap": np.exp(rng.normal(23.0, 1.2, n)),
            "sector": np.tile(sectors, len(dates)),
            "adv": np.exp(rng.normal(16.0, 1.0, n)),
            "price": np.exp(rng.normal(4.0, 0.6, n)),
            "in_universe": mask.ravel(),
        },
        index=idx,
    )


# --------------------------------------------------------------------- #
# Corruption fixtures — each introduces exactly one defect               #
# --------------------------------------------------------------------- #


def corrupt_lookahead(panel: Panel, feature: int = 5, lead: int = 0) -> Panel:
    """Leak the future: set a feature equal to a label it should only predict.

    `lead=0` sets feature[t] = label[t] — the feature IS the answer. Detected by
    the implausible-IC screen, because no honest feature has |IC| near 1.

    `lead=1` sets feature[t] = label[t+1] — the feature peeks one day past its
    own date. This is the realistic version: a `.shift(-1)` where a `.shift(1)`
    was meant. It is invisible to the implausible-IC screen (its contemporaneous
    IC is near zero) and invisible to any test that only looks at lag 0. It is
    caught by delaying the features, which is precisely what spec section 4.7's
    `test_shifted_features_lose_signal` does: a feature that predicts BETTER
    when delivered late was reading ahead.

    The no-look-ahead tests must FAIL on this panel. If they pass, they are not
    testing anything.
    """
    if lead < 0:
        raise ValueError(f"lead must be >= 0, got {lead}")
    features = panel.features.copy()
    if lead == 0:
        features[:, :, feature] = panel.labels
    else:
        features[:-lead, :, feature] = panel.labels[lead:]
        features[-lead:, :, feature] = np.nan
    return _replace(panel, features=features)


def corrupt_misaligned_labels(panel: Panel, shift: int = 1) -> Panel:
    """Slide the label axis against the feature axis by `shift` days.

    `shift=+1` gives labels[t] = old_labels[t+1]: the label is one day FURTHER
    forward than the feature was built to predict. Not leakage — it destroys
    signal — but it is the same off-by-one, and the lead-lag IC profile pins the
    direction, so both signs are worth having.

    `shift=-1` gives labels[t] = old_labels[t-1]: the label is a return already
    realized by dates[t]. This one IS leakage, and it is the direction that
    flatters a backtest.

    (Session 1 shipped this fixture with a docstring describing the negative
    case and code implementing the positive one. Both directions now exist; see
    NOTES.md, Session 2.)
    """
    if shift == 0:
        raise ValueError("shift must be non-zero")
    labels = panel.labels.copy()
    if shift > 0:
        labels[:-shift] = labels[shift:]
        labels[-shift:] = np.nan
    else:
        k = -shift
        labels[k:] = labels[:-k]
        labels[:k] = np.nan
    return _replace(panel, labels=labels)


def corrupt_survivorship(panel: Panel, drop_fraction: float = 0.2, seed: int = 0) -> Panel:
    """Remove names that are absent at the END of the sample from the ENTIRE
    history — i.e. keep only survivors. The survivorship test must catch this.
    """
    rng = np.random.default_rng(seed)
    mask = panel.mask.copy()
    n_drop = max(int(panel.n_tickers * drop_fraction), 1)
    doomed = rng.choice(panel.n_tickers, size=n_drop, replace=False)
    mask[:, doomed] = False
    mask[:, 0] = True
    return _replace(panel, mask=mask)


def corrupt_survivorship_drop_delisted(panel: Panel) -> Panel:
    """Scrub every delisted name from the ENTIRE history — real survivorship bias.

    Distinct from `corrupt_survivorship`, which removes an arbitrary subset. This
    one removes exactly the names that left the universe, which is the bias a
    naive "download today's index members and pull their history" pipeline
    produces. Requires a panel built with `n_delistings > 0`.
    """
    delisted = panel.attrs.get("delisted") or {}
    if not delisted:
        raise ValueError(
            "panel has no delistings to scrub — build it with make_synthetic_panel("
            "n_delistings=...) or the corruption is a no-op and the test proves nothing"
        )
    order = {t: i for i, t in enumerate(panel.tickers.tolist())}
    cols = [order[t] for t in delisted]

    mask = panel.mask.copy()
    features = panel.features.copy()
    labels = panel.labels.copy()
    mask[:, cols] = False
    features[:, cols, :] = np.nan
    labels[:, cols] = np.nan
    mask[:, 0] = True
    return _replace(panel, mask=mask, features=features, labels=labels)


def corrupt_regime_shift(
    panel: Panel,
    from_idx: int,
    scale: float = 4.0,
    shift: float = 2.0,
) -> Panel:
    """Rescale and re-centre features from `from_idx` onward.

    Not leakage on its own — it is the condition that makes leakage visible. If
    the later span is distributed exactly like the training span, fitting the
    normalizer on everything produces almost the same statistics as fitting on
    train alone, and a test comparing the two outputs passes whether or not the
    code leaks. A regime break gives the leakage something to move.
    """
    features = panel.features.copy()
    features[from_idx:] = (features[from_idx:] * scale + shift).astype(np.float32)
    return _replace(panel, features=features)


def corrupt_market_nan(panel: Panel, date_idx: int = 10, col: int = 0) -> Panel:
    market = panel.market.copy()
    market[date_idx, col] = np.nan
    return _replace(panel, market=market)


def _replace(panel: Panel, **changes: object) -> Panel:
    """Rebuild a Panel with substitutions, re-running all validation."""
    fields = {
        "dates": panel.dates,
        "tickers": panel.tickers,
        "features": panel.features,
        "feature_names": panel.feature_names,
        "market": panel.market,
        "market_names": panel.market_names,
        "labels": panel.labels,
        "mask": panel.mask,
        "label_horizon": panel.label_horizon,
        "metadata": panel.metadata,
        "attrs": panel.attrs,
    }
    fields.update(changes)
    return Panel(**fields)  # type: ignore[arg-type]
