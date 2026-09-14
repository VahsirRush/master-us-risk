"""Phase 6 — covariance, specific risk, and the bias-test harness.

Pure synthetic throughout: nothing here loads the real panel, so none of it
carries `pytestmark = pytest.mark.heavy` and all of it runs beside a live
training job. That is deliberate — these are the tests that would catch a
regression in the risk-forecast path, and Phase 7 will be estimating while
they need to fire.

The load-bearing tests are `test_bias_statistic_detects_*` (the gate can
actually fail) and `test_missing_names_do_not_inflate_predicted_vol` (the
harness bug that produced a false gate failure in Session 14).
"""

from __future__ import annotations

import numpy as np
import pytest

from master_us.risk.bias_tests import (
    BiasResult,
    bias_statistic,
    cap_weighted_market,
    factor_mimicking_portfolios,
    gate_passed,
    ledoit_wolf_covariance,
    portfolio_vol,
    random_portfolios,
    run_bias_tests,
    sample_covariance,
)
from master_us.risk.covariance import (
    alpha_for,
    apply_eigen_correction,
    correlation_from,
    ewma_series,
    factor_covariance,
    newey_west_variance,
    specific_risk,
)

# --------------------------------------------------------------------- #
# EWMA mechanics                                                         #
# --------------------------------------------------------------------- #


def test_alpha_matches_requested_halflife() -> None:
    """After `hl` periods a weight must have decayed to exactly one half."""
    for hl in (10.0, 40.0, 90.0):
        a = alpha_for(hl)
        assert (1.0 - a) ** hl == pytest.approx(0.5)


def test_alpha_rejects_nonpositive_halflife() -> None:
    with pytest.raises(ValueError, match="positive"):
        alpha_for(0)


def test_ewma_recovers_a_known_covariance() -> None:
    rng = np.random.default_rng(0)
    true = np.array([[4.0, 1.2], [1.2, 1.0]])
    r = rng.multivariate_normal([0, 0], true, size=20000)
    est = ewma_series(r, halflife=2000, min_periods=100)[-1]
    assert est == pytest.approx(true, abs=0.2)


def test_ewma_forecast_uses_only_data_through_t() -> None:
    """out[t] must not move when a LATER observation changes.

    This is the property the whole bias test rests on; if it fails, the gate
    is in-sample and its number is meaningless.
    """
    rng = np.random.default_rng(1)
    r = rng.normal(size=(300, 3))
    a = ewma_series(r, 40, 50)
    r2 = r.copy()
    r2[250] += 100.0
    b = ewma_series(r2, 40, 50)
    assert np.allclose(a[:250], b[:250], equal_nan=True)
    assert not np.allclose(a[250], b[250])


# --------------------------------------------------------------------- #
# the two shortcuts §12 names                                            #
# --------------------------------------------------------------------- #


def test_separate_halflives_are_genuinely_separate() -> None:
    """Guards against the single-half-life shortcut regressing back in.

    Built so the two half-lives MUST produce different answers: correlation
    flips sign halfway through the sample, so a fast correlation estimate
    and a slow one cannot agree.
    """
    rng = np.random.default_rng(2)
    n = 2000
    first = rng.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], size=n // 2)
    second = rng.multivariate_normal([0, 0], [[1, -0.9], [-0.9, 1]], size=n // 2)
    r = np.vstack([first, second])

    fast = factor_covariance(r, ("a", "b"), hl_vol=20, hl_corr=20,
                             nw_lags=0, eigen_adjust=False, min_periods=100).cov[-1]
    slow = factor_covariance(r, ("a", "b"), hl_vol=20, hl_corr=400,
                             nw_lags=0, eigen_adjust=False, min_periods=100).cov[-1]
    assert correlation_from(fast)[0, 1] != pytest.approx(correlation_from(slow)[0, 1], abs=0.1)
    # variances come from the SAME hl_vol, so they must agree
    assert np.diagonal(fast) == pytest.approx(np.diagonal(slow), rel=1e-9)


def test_newey_west_inflates_variance_under_positive_autocorrelation() -> None:
    """Serially correlated returns need more variance, not the same."""
    rng = np.random.default_rng(3)
    n = 3000
    e = rng.normal(size=n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.5 * x[i - 1] + e[i]  # strongly positively autocorrelated
    r = x[:, None]

    plain = newey_west_variance(r, 40, nw_lags=0, min_periods=100)[-1, 0]
    nw = newey_west_variance(r, 40, nw_lags=5, min_periods=100)[-1, 0]
    assert nw > plain


def test_newey_west_never_returns_a_nonpositive_variance() -> None:
    """Strong negative autocorrelation can drive the adjustment below zero."""
    rng = np.random.default_rng(4)
    n = 2000
    e = rng.normal(size=n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = -0.9 * x[i - 1] + e[i]
    nw = newey_west_variance(x[:, None], 40, nw_lags=5, min_periods=100)
    finite = nw[np.isfinite(nw)]
    assert (finite > 0).all()


def test_covariance_is_positive_semidefinite() -> None:
    rng = np.random.default_rng(5)
    r = rng.normal(size=(1500, 6)) @ rng.normal(size=(6, 6))
    fc = factor_covariance(r, tuple("abcdef"), min_periods=200)
    last = fc.cov[-1]
    evals = np.linalg.eigvalsh(last)
    assert evals.min() > -1e-10


# --------------------------------------------------------------------- #
# eigenfactor adjustment                                                 #
# --------------------------------------------------------------------- #


def test_eigen_correction_inflates_underforecast_directions() -> None:
    """lambda < 1 means under-forecast, so the correction must raise it."""
    # eigh returns eigenvalues ASCENDING, so lam[0] pairs with the SMALLEST
    # eigenvalue — which is the realistic case: a sample covariance
    # under-forecasts its minimum-variance directions most.
    cov = np.array([[[4.0, 0.0], [0.0, 1.0]]])
    lam = np.array([0.5, 1.0])
    out = apply_eigen_correction(cov, lam)
    d_in = np.linalg.eigvalsh(cov[0])
    d_out = np.linalg.eigvalsh(out[0])
    assert d_out.min() > d_in.min()
    assert d_out.min() == pytest.approx(d_in.min() / 0.5**2)
    assert d_out.max() == pytest.approx(d_in.max())  # unbiased rank untouched


def test_eigen_correction_is_identity_when_unbiased() -> None:
    rng = np.random.default_rng(6)
    a = rng.normal(size=(4, 4))
    cov = (a @ a.T)[None, :, :]
    out = apply_eigen_correction(cov, np.ones(4))
    assert out[0] == pytest.approx(cov[0], abs=1e-10)


# --------------------------------------------------------------------- #
# specific risk                                                          #
# --------------------------------------------------------------------- #


def test_specific_risk_tracks_a_known_level() -> None:
    rng = np.random.default_rng(7)
    t, n = 1200, 40
    true_vol = np.linspace(0.01, 0.05, n)
    resid = rng.normal(size=(t, n)) * true_vol
    sr = specific_risk(
        resid,
        size=np.zeros((t, n)),
        leverage=np.zeros((t, n)),
        industry=np.array(["A"] * n, dtype=object),
        halflife=120,
        shrink=False,
        min_periods=100,
    )
    est = np.sqrt(sr.variance[-1])
    assert np.corrcoef(est, true_vol)[0, 1] > 0.9


def test_specific_risk_shrinkage_fills_sparse_names() -> None:
    """A name with almost no history still gets a forecast, from its peers."""
    # 60 names, not 30: `structural_specific_vol` refuses to fit on fewer
    # than 30 usable observations, and excluding the sparse name from a
    # 30-name cross-section leaves 29.
    rng = np.random.default_rng(8)
    t, n = 400, 60
    resid = rng.normal(size=(t, n)) * 0.02
    resid[:, 0] = np.nan
    resid[-5:, 0] = rng.normal(size=5) * 0.02
    sr = specific_risk(
        resid,
        size=np.tile(np.linspace(-1, 1, n), (t, 1)),
        leverage=np.zeros((t, n)),
        industry=np.array(["A"] * n, dtype=object),
        halflife=60,
        shrink=True,
        min_periods=60,
    )
    assert np.isfinite(sr.variance[-1, 0])
    assert sr.variance[-1, 0] > 0


# --------------------------------------------------------------------- #
# the bias statistic itself                                              #
# --------------------------------------------------------------------- #


def test_bias_statistic_is_one_when_perfectly_calibrated() -> None:
    rng = np.random.default_rng(9)
    vol = 0.02
    realized = rng.normal(size=50000) * vol
    assert bias_statistic(realized, np.full(50000, vol)) == pytest.approx(1.0, abs=0.02)


def test_bias_statistic_detects_underforecasting() -> None:
    """Half the true vol must read ~2.0, not ~1.0 — the gate must be able to fail."""
    rng = np.random.default_rng(10)
    realized = rng.normal(size=50000) * 0.02
    assert bias_statistic(realized, np.full(50000, 0.01)) == pytest.approx(2.0, abs=0.05)


def test_bias_statistic_detects_overforecasting() -> None:
    rng = np.random.default_rng(11)
    realized = rng.normal(size=50000) * 0.01
    assert bias_statistic(realized, np.full(50000, 0.02)) == pytest.approx(0.5, abs=0.02)


def test_gate_needs_a_majority_not_a_single_portfolio() -> None:
    inside = BiasResult("x", {"random": np.array([1.0] * 6 + [3.0] * 4)})
    outside = BiasResult("y", {"random": np.array([1.0] * 4 + [3.0] * 6)})
    assert gate_passed(inside)
    assert not gate_passed(outside)


# --------------------------------------------------------------------- #
# portfolio machinery                                                    #
# --------------------------------------------------------------------- #


def test_portfolio_vol_matches_the_explicit_form() -> None:
    """The collapsed form must equal w'(XFX' + D)w computed directly."""
    rng = np.random.default_rng(12)
    n, k = 40, 4
    x = rng.normal(size=(n, k))
    a = rng.normal(size=(k, k))
    f = a @ a.T
    d = np.abs(rng.normal(size=n)) * 1e-4
    w = rng.normal(size=(3, n))

    got = portfolio_vol(w, x, f, d)
    full = x @ f @ x.T + np.diag(d)
    want = np.sqrt(np.einsum("pi,ij,pj->p", w, full, w))
    assert got == pytest.approx(want, rel=1e-10)


def test_random_portfolios_are_dollar_neutral() -> None:
    w = random_portfolios(200, n_portfolios=50, seed=0)
    assert np.abs(w.sum(axis=1)).max() < 1e-12
    assert np.abs(w).sum(axis=1) == pytest.approx(np.ones(50), rel=1e-9)


def test_factor_mimicking_portfolio_has_unit_exposure() -> None:
    rng = np.random.default_rng(13)
    n, k = 200, 3
    x = rng.normal(size=(n, k))
    mcap = np.exp(rng.normal(20, 1, n))
    w = factor_mimicking_portfolios(x, mcap)
    assert w @ x == pytest.approx(np.eye(k), abs=1e-8)


def test_cap_weighted_market_excludes_secondary_classes() -> None:
    mcap = np.array([100.0, 100.0, 50.0])
    primary = np.array([True, False, True])
    w = cap_weighted_market(mcap, primary)
    assert w[1] == 0.0
    assert w.sum() == pytest.approx(1.0)


def test_missing_names_do_not_inflate_predicted_vol() -> None:
    """Regression test for the Session 14 harness bug.

    Only some names have a realized return in any period. Charging a
    portfolio's predicted variance for names it cannot be judged on inflated
    predicted vol by ~1/coverage and produced a FALSE gate failure at 49.9%.
    Predicted and realized must describe the same holdings.
    """
    rng = np.random.default_rng(14)
    t, n, k = 400, 60, 3
    exposures = np.tile(rng.normal(size=(n, k)), (t, 1, 1))
    f = np.eye(k) * 1e-4
    covs = {"m": np.tile(f, (t, 1, 1))}
    spec = np.full((t, n), 4e-4)
    mcap = np.tile(np.exp(rng.normal(20, 1, n)), (t, 1))
    primary = np.ones(n, dtype=bool)

    returns = rng.normal(size=(t, n)) * 0.02
    full = run_bias_tests(
        np.zeros((t, k)), exposures, returns, spec, mcap, primary, covs, start=50, seed=0
    )
    # now hide half the names entirely — the same portfolios, less coverage
    holed = returns.copy()
    holed[:, ::2] = np.nan
    partial = run_bias_tests(
        np.zeros((t, k)), exposures, holed, spec, mcap, primary, covs, start=50, seed=0
    )

    a = np.nanmedian(full["m"].by_family["random"])
    b = np.nanmedian(partial["m"].by_family["random"])
    # coverage changes which names are held, so these need not be identical,
    # but a systematic deflation would mean predicted vol is being charged
    # for names that contribute no return
    assert abs(a - b) < 0.15, f"coverage changed the bias statistic: {a:.3f} -> {b:.3f}"


# --------------------------------------------------------------------- #
# benchmarks                                                             #
# --------------------------------------------------------------------- #


def test_sample_covariance_is_only_backward_looking() -> None:
    rng = np.random.default_rng(15)
    r = rng.normal(size=(600, 4))
    a = sample_covariance(r, window=252)
    r2 = r.copy()
    r2[500] += 50.0
    b = sample_covariance(r2, window=252)
    assert np.allclose(a[:500], b[:500], equal_nan=True)


def test_ledoit_wolf_shrinks_toward_identity() -> None:
    """With fewer observations the estimate must sit closer to the target."""
    rng = np.random.default_rng(16)
    k = 8
    a = rng.normal(size=(k, k))
    true = a @ a.T
    r = rng.multivariate_normal(np.zeros(k), true, size=800)

    lw = ledoit_wolf_covariance(r, window=300)[-1]
    sc = sample_covariance(r, window=300)[-1]
    target = np.trace(sc) / k * np.eye(k)
    off_lw = np.abs(lw - np.diag(np.diagonal(lw))).sum()
    off_sc = np.abs(sc - np.diag(np.diagonal(sc))).sum()
    assert off_lw < off_sc
    assert np.abs(lw - target).sum() < np.abs(sc - target).sum()
