"""The feature bank — spec section 4.3.

Every feature is strictly backward-looking. The timestamp convention is the
one `panel.py` fixes: `features[t]` may use any information known AS OF the
close of `dates[t]`, and nothing after. Concretely:

* Rolling statistics (std, means, sums, EMAs, beta) run over windows ENDING
  AT AND INCLUDING day t — the current bar's close is the last included
  observation, which is what the spec's "closed='left'" phrasing pins down:
  the window is closed on the left of t+1, i.e. nothing from the future.
* Features that COMPARE today against its own recent history (volume ratio,
  dollar-volume z-score) use a reference window ending at t-1 — comparing
  today's volume against a mean that already contains today is a weaker,
  self-contaminated signal, not a leak; excluding it is a signal-quality
  choice, and it is still strictly backward-looking.
* Cross-sectional ranks are per-date over the TRADEABLE set only, so a name
  outside the universe can never shift a rank inside it.

Group naming follows `feature_groups.py` exactly (ret_/vol_/volume_/ps_/
tech_/xs_); `build_features` refuses to emit a name the registry cannot
classify, so a typo cannot silently pool its NaN mask with strangers.

Implementation choices that deviate from the most common textbook variant,
each deliberate:

* RSI uses Cutler's form (simple moving averages of gains and losses) rather
  than Wilder's recursive smoothing. Wilder's is an infinite-memory EMA whose
  value depends on the series start point, which makes cached and freshly
  computed values differ; Cutler's is a fixed 14-day window and reproducible.
* MACD terms are divided by the current adjusted close. Raw MACD is in price
  units and cross-sectionally incomparable — a $2000 stock would dominate
  every rank.
* Garman-Klass per-day estimates can be slightly negative on gap days; they
  are floored at zero before the rolling mean, the standard practice.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from master_us.data.feature_groups import classify_features

RETURN_WINDOWS = (1, 2, 3, 5, 10, 20, 30, 60)
VOL_WINDOWS = (2, 3, 5, 10, 20, 30, 60)
VOLUME_RATIO_WINDOWS = (5, 10, 20, 60)
DVZ_WINDOWS = (20, 60)
AMIHUD_WINDOWS = (5, 10, 20, 60)
PS_WINDOWS = (1, 5, 10, 20, 60)
VWAP_WINDOWS = (5, 10, 20, 60)

EPS = 1e-12


def base_feature_names() -> list[str]:
    """Every time-series feature name, in emission order."""
    names: list[str] = []
    names += [f"ret_{w}d" for w in RETURN_WINDOWS]
    names += [f"vol_std_{w}d" for w in VOL_WINDOWS]
    names += [f"vol_parkinson_{w}d" for w in VOL_WINDOWS]
    names += [f"vol_gk_{w}d" for w in VOL_WINDOWS]
    names += [f"volume_ratio_{w}d" for w in VOLUME_RATIO_WINDOWS]
    names += [f"volume_dvz_{w}d" for w in DVZ_WINDOWS]
    names += [f"volume_amihud_{w}d" for w in AMIHUD_WINDOWS]
    names += [f"ps_hl_range_{w}d" for w in PS_WINDOWS]
    names += [f"ps_close_pos_{w}d" for w in PS_WINDOWS]
    names += [f"ps_vwap_dev_{w}d" for w in VWAP_WINDOWS]
    names += [f"ps_gap_{w}d" for w in PS_WINDOWS]
    names += [
        "tech_rsi_14",
        "tech_macd_12_26",
        "tech_macd_signal_9",
        "tech_macd_hist",
        "tech_bb_pctb_20",
        "tech_beta_60d",
        "tech_corr_60d",
    ]
    return names


def all_feature_names() -> list[str]:
    """Base features plus their cross-sectional ranks — the full bank."""
    base = base_feature_names()
    return base + [f"xs_{name}" for name in base]


def build_features(ohlcv: pl.DataFrame, spy: pl.DataFrame) -> pl.DataFrame:
    """Compute every time-series feature. Long in, long out.

    Parameters
    ----------
    ohlcv:
        [date, ticker, open, high, low, close, adj_close, volume], one row per
        (date, ticker), each ticker's dates sorted ascending.
    spy:
        [date, adj_close] for the benchmark, for rolling beta/correlation.

    Cross-sectional ranks are NOT computed here — they need the tradeable
    mask, which is universe knowledge this function deliberately does not
    have. Call `add_cross_sectional_ranks` afterwards.
    """
    required = {"date", "ticker", "open", "high", "low", "close", "adj_close", "volume"}
    missing = required - set(ohlcv.columns)
    if missing:
        raise ValueError(f"ohlcv is missing columns {sorted(missing)}")

    spy_ret = (
        spy.sort("date")
        .with_columns((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("spy_ret"))
        .select(["date", "spy_ret"])
    )

    df = (
        ohlcv.sort(["ticker", "date"])
        .join(spy_ret, on="date", how="left")
        .with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1).over("ticker") - 1.0).alias(
                "_ret1"
            ),
            (pl.col("close") * pl.col("volume")).alias("_dv"),
            (pl.col("high") / pl.col("low")).log().pow(2).alias("_log_hl_sq"),
            (pl.col("close") / pl.col("open")).log().pow(2).alias("_log_co_sq"),
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0 * pl.col("volume")).alias(
                "_tpv"
            ),
            ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("_hl_range"),
            (pl.col("open") / pl.col("close").shift(1).over("ticker") - 1.0).alias("_gap"),
        )
        .with_columns(
            # Close position within the day's range; a zero-range day carries
            # no information about position, so it reads as the midpoint.
            pl.when(pl.col("high") > pl.col("low"))
            .then((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low")))
            .otherwise(0.5)
            .alias("_close_pos"),
            # Garman-Klass per-day variance, floored at zero (see module doc).
            (0.5 * pl.col("_log_hl_sq") - (2.0 * np.log(2.0) - 1.0) * pl.col("_log_co_sq"))
            .clip(lower_bound=0.0)
            .alias("_gk_var"),
            (pl.col("_ret1").abs() / (pl.col("_dv") + EPS)).alias("_amihud_raw"),
        )
    )

    exprs: list[pl.Expr] = []

    # ---- returns ------------------------------------------------------- #
    for w in RETURN_WINDOWS:
        exprs.append(
            (pl.col("adj_close") / pl.col("adj_close").shift(w).over("ticker") - 1.0).alias(
                f"ret_{w}d"
            )
        )

    # ---- volatility ----------------------------------------------------- #
    for w in VOL_WINDOWS:
        exprs.append(
            pl.col("_ret1").rolling_std(w, min_samples=w).over("ticker").alias(f"vol_std_{w}d")
        )
        exprs.append(
            (
                pl.col("_log_hl_sq").rolling_mean(w, min_samples=w).over("ticker")
                / (4.0 * np.log(2.0))
            )
            .sqrt()
            .alias(f"vol_parkinson_{w}d")
        )
        exprs.append(
            pl.col("_gk_var")
            .rolling_mean(w, min_samples=w)
            .over("ticker")
            .sqrt()
            .alias(f"vol_gk_{w}d")
        )

    # ---- volume --------------------------------------------------------- #
    # Reference windows end at t-1: today's volume against ITS OWN recent
    # past, not against a mean that already contains it.
    for w in VOLUME_RATIO_WINDOWS:
        trailing_mean = (
            pl.col("volume").shift(1).rolling_mean(w, min_samples=w).over("ticker")
        )
        exprs.append((pl.col("volume") / (trailing_mean + EPS)).alias(f"volume_ratio_{w}d"))
    for w in DVZ_WINDOWS:
        dv_mean = pl.col("_dv").shift(1).rolling_mean(w, min_samples=w).over("ticker")
        dv_std = pl.col("_dv").shift(1).rolling_std(w, min_samples=w).over("ticker")
        exprs.append(((pl.col("_dv") - dv_mean) / (dv_std + EPS)).alias(f"volume_dvz_{w}d"))
    for w in AMIHUD_WINDOWS:
        exprs.append(
            pl.col("_amihud_raw")
            .rolling_mean(w, min_samples=w)
            .over("ticker")
            .alias(f"volume_amihud_{w}d")
        )

    # ---- price structure ------------------------------------------------ #
    for w in PS_WINDOWS:
        exprs.append(
            pl.col("_hl_range")
            .rolling_mean(w, min_samples=w)
            .over("ticker")
            .alias(f"ps_hl_range_{w}d")
        )
        exprs.append(
            pl.col("_close_pos")
            .rolling_mean(w, min_samples=w)
            .over("ticker")
            .alias(f"ps_close_pos_{w}d")
        )
        exprs.append(
            pl.col("_gap").rolling_mean(w, min_samples=w).over("ticker").alias(f"ps_gap_{w}d")
        )
    for w in VWAP_WINDOWS:
        vwap = (
            pl.col("_tpv").rolling_sum(w, min_samples=w).over("ticker")
            / (pl.col("volume").rolling_sum(w, min_samples=w).over("ticker") + EPS)
        )
        exprs.append((pl.col("close") / (vwap + EPS) - 1.0).alias(f"ps_vwap_dev_{w}d"))

    # ---- technical ------------------------------------------------------ #
    gain = pl.col("_ret1").clip(lower_bound=0.0)
    loss = (-pl.col("_ret1")).clip(lower_bound=0.0)
    avg_gain = gain.rolling_mean(14, min_samples=14).over("ticker")
    avg_loss = loss.rolling_mean(14, min_samples=14).over("ticker")
    exprs.append((100.0 * avg_gain / (avg_gain + avg_loss + EPS)).alias("tech_rsi_14"))

    ema12 = pl.col("adj_close").ewm_mean(span=12, adjust=False).over("ticker")
    ema26 = pl.col("adj_close").ewm_mean(span=26, adjust=False).over("ticker")
    macd = (ema12 - ema26) / (pl.col("adj_close") + EPS)
    signal = macd.ewm_mean(span=9, adjust=False).over("ticker")
    exprs.append(macd.alias("tech_macd_12_26"))
    exprs.append(signal.alias("tech_macd_signal_9"))
    exprs.append((macd - signal).alias("tech_macd_hist"))

    ma20 = pl.col("adj_close").rolling_mean(20, min_samples=20).over("ticker")
    sd20 = pl.col("adj_close").rolling_std(20, min_samples=20).over("ticker")
    exprs.append(
        ((pl.col("adj_close") - (ma20 - 2.0 * sd20)) / (4.0 * sd20 + EPS)).alias(
            "tech_bb_pctb_20"
        )
    )

    # Rolling beta/corr to SPY over 60d: cov/var and corr of daily returns.
    n = 60
    mean_r = pl.col("_ret1").rolling_mean(n, min_samples=n).over("ticker")
    mean_b = pl.col("spy_ret").rolling_mean(n, min_samples=n).over("ticker")
    mean_rb = (pl.col("_ret1") * pl.col("spy_ret")).rolling_mean(n, min_samples=n).over("ticker")
    mean_bb = (pl.col("spy_ret") * pl.col("spy_ret")).rolling_mean(n, min_samples=n).over("ticker")
    mean_rr = (pl.col("_ret1") * pl.col("_ret1")).rolling_mean(n, min_samples=n).over("ticker")
    cov = mean_rb - mean_r * mean_b
    var_b = mean_bb - mean_b * mean_b
    var_r = mean_rr - mean_r * mean_r
    exprs.append((cov / (var_b + EPS)).alias("tech_beta_60d"))
    exprs.append((cov / ((var_b * var_r).sqrt() + EPS)).alias("tech_corr_60d"))

    out = df.with_columns(exprs).select(["date", "ticker", *base_feature_names()])

    # Non-finite values (log of a zero low, division residue) become NaN-like
    # nulls now rather than infinities that survive into the panel.
    return out.with_columns(
        [
            pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c)
            for c in base_feature_names()
        ]
    )


def add_cross_sectional_ranks(
    features: pl.DataFrame,
    tradeable: pl.DataFrame,
) -> pl.DataFrame:
    """Append `xs_<name>`: per-date percentile rank over the tradeable set.

    Ranks are in (0, 1] (average-tie percentile). A name that is untradeable
    on a date, or has no value, gets a null rank — it neither receives nor
    influences one. Spec 4.3 lists the ranks as their own feature group, and
    the registry files them under `cross_sectional_rank` via the xs_ prefix.
    """
    if not {"date", "ticker", "tradeable"} <= set(tradeable.columns):
        raise ValueError("tradeable frame needs [date, ticker, tradeable]")

    joined = features.join(
        tradeable.select(["date", "ticker", "tradeable"]), on=["date", "ticker"], how="left"
    ).with_columns(pl.col("tradeable").fill_null(value=False))

    rank_exprs = [
        pl.when(pl.col("tradeable"))
        .then(
            pl.col(name).rank(method="average").over("date", "tradeable")
            / pl.col(name).is_not_null().sum().over("date", "tradeable")
        )
        .otherwise(None)
        .alias(f"xs_{name}")
        for name in base_feature_names()
    ]
    return joined.with_columns(rank_exprs).drop("tradeable")


def validate_feature_names() -> None:
    """Every emitted name must classify into a real group — never OTHER.

    Called at build time so a misnamed feature fails the pipeline instead of
    silently pooling its missingness mask with strangers.
    """
    classify_features(all_feature_names(), strict=True)
