"""Phase 5 — descriptors, standardization, and factor returns.

The load-bearing tests here are the synthetic-recovery ones
(`test_recovers_known_factor_returns`, `test_industry_constraint_holds`): a
cross-sectional regression that cannot recover factor returns it generated
itself is broken in a way no amount of plausible-looking output on real data
would reveal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from master_us.risk.descriptors import (
    fundamental_asof,
    fundamental_lagged,
    momentum_12_1,
    resolve_concept,
)
from master_us.risk.factor_returns import estimate_factor_returns, estimate_period
from master_us.risk.industry import UNKNOWN, sic_to_industry
from master_us.risk.sanity import run_checks
from master_us.risk.standardize import (
    demean_cap_weighted,
    winsorize,
    zscore_within_industry,
)

# --------------------------------------------------------------------- #
# industry                                                               #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("sic", "expected"),
    [
        (4911, "Utilities"),          # electric services
        (6798, "Real Estate"),        # REIT
        (7372, "Information Technology"),  # prepackaged software
        (2834, "Health Care"),        # pharmaceutical preparations
        (3711, "Consumer Discretionary"),  # motor vehicles
        (3721, "Industrials"),        # aircraft — override inside the 3700s
        (6022, "Financials"),         # state commercial banks
        (1311, "Energy"),             # crude petroleum
        (0, UNKNOWN),
        (None, UNKNOWN),
    ],
)
def test_sic_mapping(sic: int | None, expected: str) -> None:
    assert sic_to_industry(sic) == expected


def test_aerospace_override_beats_the_broad_range() -> None:
    """3721 sits inside the 3700-3799 consumer block but is Industrials.

    Guards the ordering in `sic_to_industry`: overrides must be tested
    before the broad ranges, or aerospace silently becomes a consumer name.
    """
    assert sic_to_industry(3711) == "Consumer Discretionary"
    assert sic_to_industry(3721) == "Industrials"
    assert sic_to_industry(3760) == "Industrials"


# --------------------------------------------------------------------- #
# standardization — spec §9.2                                            #
# --------------------------------------------------------------------- #


def test_winsorize_clips_at_three_sigma() -> None:
    x = np.concatenate([np.zeros(100), [1000.0]])
    w = winsorize(x, 3.0)
    assert w.max() < 1000.0
    assert np.isfinite(w).all()


def test_winsorize_ignores_nan() -> None:
    x = np.array([1.0, 2.0, np.nan, 3.0, 100.0])
    w = winsorize(x)
    assert np.isnan(w[2])
    assert np.isfinite(w[[0, 1, 3, 4]]).all()


def test_zscore_is_within_industry_not_global() -> None:
    """The distinction the spec calls out as the common shortcut.

    Two industries with different means must each be centred on their own
    mean; a global z-score would leave one wholly positive and the other
    wholly negative.
    """
    x = np.array([10.0, 12.0, 14.0, 100.0, 102.0, 104.0])
    ind = np.array(["A"] * 3 + ["B"] * 3, dtype=object)
    z = zscore_within_industry(x, ind, min_members=3)

    assert abs(z[:3].mean()) < 1e-12
    assert abs(z[3:].mean()) < 1e-12
    # a global z-score would separate the groups entirely; this must not
    assert np.sign(z[:3]).tolist() == np.sign(z[3:]).tolist()


def test_small_industry_falls_back_to_global() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 50.0])
    ind = np.array(["A", "A", "A", "A", "B"], dtype=object)
    z = zscore_within_industry(x, ind, min_members=3)
    assert np.isfinite(z).all()
    assert z[4] > z[:4].max()  # the lone name keeps its global extremity


def test_zero_dispersion_industry_gives_zeros_not_infinities() -> None:
    x = np.array([5.0, 5.0, 5.0, 5.0])
    ind = np.array(["A"] * 4, dtype=object)
    z = zscore_within_industry(x, ind, min_members=2)
    assert np.isfinite(z).all()
    assert np.allclose(z, 0.0)


def test_cap_weighted_demean_is_exact() -> None:
    x = np.array([1.0, 2.0, 3.0])
    mcap = np.array([100.0, 10.0, 1.0])
    d = demean_cap_weighted(x, mcap)
    assert abs(float((mcap * d).sum() / mcap.sum())) < 1e-12


def test_demean_excludes_secondary_share_classes() -> None:
    """A dual-class issuer must not be weighted twice in its own mean."""
    x = np.array([1.0, 1.0, -1.0])
    mcap = np.array([100.0, 100.0, 100.0])  # first two are the same company
    primary = np.array([True, False, True])
    d = demean_cap_weighted(x, mcap, primary)
    w = mcap * primary
    assert abs(float((w * d).sum() / w.sum())) < 1e-12


# --------------------------------------------------------------------- #
# point-in-time discipline                                               #
# --------------------------------------------------------------------- #


def _facts() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "cik": [1, 1, 1],
            "filed": pd.to_datetime(["2020-02-15", "2020-05-15", "2020-08-15"]),
            "period_end": pd.to_datetime(["2019-12-31", "2020-03-31", "2020-06-30"]),
            "tag": ["Assets"] * 3,
            "value": [100.0, 200.0, 300.0],
        }
    ).with_columns(pl.col("filed").cast(pl.Datetime("ns")))


def test_fundamental_asof_never_uses_unfiled_data() -> None:
    """Rule 1: a value is knowable only from its `filed` date onward.

    The day before a filing must still carry the previous filing's value,
    never the one about to be published.
    """
    grid = pl.DataFrame(
        {
            "date": pd.to_datetime(["2020-02-14", "2020-02-15", "2020-05-14", "2020-05-15"]),
            "ticker": ["AAA"] * 4,
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ns")))
    out = fundamental_asof(grid, _facts(), "assets")["assets"].to_list()
    assert out[0] is None      # nothing filed yet
    assert out[1] == 100.0     # available on the filing date
    assert out[2] == 100.0     # still the old one the day before the next
    assert out[3] == 200.0


def test_fundamental_asof_respects_staleness_cap() -> None:
    grid = pl.DataFrame(
        {"date": pd.to_datetime(["2025-01-01"]), "ticker": ["AAA"]}
    ).with_columns(pl.col("date").cast(pl.Datetime("ns")))
    out = fundamental_asof(grid, _facts(), "assets", tolerance_days=400)["assets"].to_list()
    assert out[0] is None  # last filing is years old — null, not a stale guess


def test_fundamental_lagged_is_also_point_in_time() -> None:
    grid = pl.DataFrame(
        {"date": pd.to_datetime(["2023-02-20"]), "ticker": ["AAA"]}
    ).with_columns(pl.col("date").cast(pl.Datetime("ns")))
    out = fundamental_lagged(grid, _facts(), "assets", years=3)
    assert out["assets_lag3y"].to_list()[0] == 100.0  # what was known in Feb 2020


def test_resolve_concept_prefers_higher_priority_tag() -> None:
    facts = pl.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "cik": [1, 1],
            "filed": pd.to_datetime(["2020-02-15"] * 2),
            "period_end": pd.to_datetime(["2019-12-31"] * 2),
            "tag": ["ProfitLoss", "NetIncomeLoss"],  # second is higher priority
            "value": [1.0, 2.0],
        }
    ).with_columns(pl.col("filed").cast(pl.Datetime("ns")))
    assert resolve_concept(facts, "net_income")["net_income"].to_list() == [2.0]


def test_momentum_skips_the_most_recent_month() -> None:
    """A spike inside the last 21 days must not enter the 12-1 window."""
    rets = np.zeros((300, 1))
    rets[290, 0] = 0.5  # inside the skipped month
    assert abs(momentum_12_1(rets)[299, 0]) < 1e-9

    rets2 = np.zeros((300, 1))
    rets2[100, 0] = 0.5  # inside the 12-1 window
    assert momentum_12_1(rets2)[299, 0] > 0.4


# --------------------------------------------------------------------- #
# factor returns — spec §9.3                                             #
# --------------------------------------------------------------------- #


def _synthetic(seed: int = 0, n: int = 300):
    rng = np.random.default_rng(seed)
    industry = np.array(rng.choice(["A", "B", "C"], n), dtype=object)
    mcap = np.exp(rng.normal(22.0, 1.2, n))
    x = np.column_stack([rng.normal(size=n), rng.normal(size=n)])
    x -= x.mean(axis=0)
    return industry, mcap, x, rng


def test_recovers_known_factor_returns() -> None:
    """Generate returns from known factor returns; the WLS must recover them.

    Noise is small relative to the factor returns so recovery should be
    tight. If this drifts, the regression, the weighting, or the constraint
    reparameterization is wrong.
    """
    industry, mcap, x, rng = _synthetic()
    true_style = np.array([0.03, -0.02])
    true_market = 0.01
    ind_names = ("A", "B", "C")
    true_ind = {"A": 0.01, "B": -0.005, "C": -0.004}
    ind_effect = np.array([true_ind[i] for i in industry])

    r = true_market + x @ true_style + ind_effect + rng.normal(0, 1e-4, len(mcap))
    fs, _fi, _spec, r2, n = estimate_period(x, r, mcap, industry, ind_names)

    assert n == len(mcap)
    assert fs[1] == pytest.approx(true_style[0], abs=2e-3)
    assert fs[2] == pytest.approx(true_style[1], abs=2e-3)
    assert r2 > 0.99


def test_industry_constraint_holds() -> None:
    """Cap-weighted industry returns must sum to zero — the §9.3 constraint.

    This is what makes the intercept the market factor rather than an
    arbitrary reference industry's return.
    """
    industry, mcap, x, rng = _synthetic(seed=3)
    ind_names = ("A", "B", "C")
    r = 0.01 + x @ np.array([0.02, 0.01]) + rng.normal(0, 0.01, len(mcap))
    _, f_ind, _, _, _ = estimate_period(x, r, mcap, industry, ind_names)

    cw = np.array([mcap[industry == j].sum() for j in ind_names])
    cw = cw / cw.sum()
    assert abs(float(cw @ f_ind)) < 1e-10


def test_specific_returns_are_orthogonal_to_exposures() -> None:
    """The WLS normal equations: sum_i w_i * x_ik * u_i = 0 for every k.

    Orthogonality holds in the WEIGHTED inner product, not the unweighted
    correlation — an unweighted Pearson correlation between exposure and
    residual is only approximately zero here (it came out 0.055 on this
    seed), and asserting on it would be testing the wrong identity with a
    tolerance tuned until it passed.
    """
    from master_us.risk.factor_returns import _weights

    industry, mcap, x, rng = _synthetic(seed=7)
    r = 0.01 + x @ np.array([0.02, -0.01]) + rng.normal(0, 0.02, len(mcap))
    _, _, spec, _, _ = estimate_period(x, r, mcap, industry, ("A", "B", "C"))
    ok = np.isfinite(spec)
    w = _weights(mcap[ok])
    scale = float(np.abs(w * spec[ok]).sum())
    for k in range(x.shape[1]):
        assert abs(float((w * x[ok, k] * spec[ok]).sum())) < 1e-8 * max(scale, 1.0)


def test_refuses_a_thin_cross_section() -> None:
    """Below MIN_NAMES the period returns NaN rather than a fitted value."""
    industry = np.array(["A"] * 5, dtype=object)
    mcap = np.full(5, 1e9)
    x = np.zeros((5, 2))
    fs, _fi, _, r2, n = estimate_period(x, np.zeros(5), mcap, industry, ("A",))
    assert n == 5
    assert np.isnan(fs).all()
    assert np.isnan(r2)


def test_missing_exposures_are_imputed_not_dropped() -> None:
    """A name missing one factor still enters, with that exposure at zero.

    Requiring every factor gated the whole cross-section on the weakest
    one and produced no regression at all before 2013 on real data.
    """
    industry, mcap, x, rng = _synthetic(seed=11)
    x = np.column_stack([x, rng.normal(size=len(mcap)), rng.normal(size=len(mcap)),
                         rng.normal(size=len(mcap)), rng.normal(size=len(mcap))])
    x[:50, 0] = np.nan  # 5 of 6 factors present — above the 0.75 fraction
    r = 0.01 + np.nan_to_num(x) @ np.full(x.shape[1], 0.01) + rng.normal(0, 1e-3, len(mcap))
    _, _, _, _, n = estimate_period(x, r, mcap, industry, ("A", "B", "C"))
    assert n == len(mcap)


def test_estimate_factor_returns_shapes_and_alignment() -> None:
    industry, mcap, x, rng = _synthetic(seed=5)
    t = 24
    dates = pd.date_range("2020-01-31", periods=t, freq="ME")
    exposures = {
        "f1": np.tile(x[:, 0], (t, 1)),
        "f2": np.tile(x[:, 1], (t, 1)),
    }
    rets = rng.normal(0, 0.05, (t, len(mcap)))
    fr = estimate_factor_returns(
        exposures, rets, np.tile(mcap, (t, 1)), industry, dates
    )
    assert fr.returns.shape == (t, 3)  # market + 2 styles
    assert fr.specific.shape == rets.shape
    assert fr.factor_names == ("market", "f1", "f2")
    assert len(fr.to_frame()) == t


def test_sanity_gate_flags_a_broken_model() -> None:
    """The gate must fail on noise, or it is not a gate.

    Pure noise has no momentum premium, no value cycle, and a market
    intercept that is not equity-like; at least one check must catch it.
    """
    rng = np.random.default_rng(0)
    t = 120
    dates = pd.date_range("2015-01-31", periods=t, freq="ME")
    names = ("market", "size", "value", "momentum", "volatility",
             "liquidity", "leverage", "growth", "quality")
    fr = type("FR", (), {})()
    from master_us.risk.factor_returns import FactorReturns

    fr = FactorReturns(
        dates=dates,
        factor_names=names,
        industry_names=(),
        returns=rng.normal(0, 0.0001, (t, len(names))),
        industry_returns=np.zeros((t, 0)),
        specific=np.zeros((t, 10)),
        r2=np.full(t, 0.25),
        n_names=np.full(t, 100, dtype=np.int64),
    )
    assert not run_checks(fr).passed
