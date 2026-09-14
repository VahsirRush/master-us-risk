"""Tests for the result contract — output-layer-spec section 2.

`distinguishable_from` gets the most attention: it is the mechanism behind
docs/project-conventions.md rule 4 ("a gap smaller than pooled seed dispersion must be reported
as not distinguishable"), and a renderer trusting a broken version would
print noise as findings.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from master_us.reporting.results import MetricValue, PhaseResult

# ------------------------------------------------------------------ #
# MetricValue                                                         #
# ------------------------------------------------------------------ #


def test_render_gross_only():
    assert MetricValue(gross=0.0342).render() == "0.0342"


def test_render_with_net_and_std():
    mv = MetricValue(gross=0.0342, net=0.0198, std=0.0061, n_seeds=5)
    assert mv.render() == "0.0342 (net 0.0198) ± 0.0061"
    assert MetricValue(gross=0.0342, std=0.0061).render() == "0.0342 ± 0.0061"


def test_render_respects_format():
    assert MetricValue(gross=0.9412).render(".2f") == "0.94"


def test_distinguishable_uses_pooled_std_not_any_nonzero_gap():
    """A gap of 0.004 with stds of 0.0058 and 0.0061 is NOISE.

    Pooled std = sqrt(0.0058² + 0.0061²) ≈ 0.00842 > 0.004. A naive
    implementation comparing against either single std (0.0058) would also
    call this noise, but one comparing against |gap| > 0 would call it real —
    which is the failure this test exists to catch.
    """
    full = MetricValue(gross=0.0341, std=0.0058, n_seeds=5)
    ablated = MetricValue(gross=0.0301, std=0.0061, n_seeds=5)
    assert not full.distinguishable_from(ablated)
    assert not ablated.distinguishable_from(full)

    pooled = math.sqrt(0.0058**2 + 0.0061**2)
    barely_real = MetricValue(gross=0.0341 - pooled - 1e-6, std=0.0061, n_seeds=5)
    assert full.distinguishable_from(barely_real)


def test_distinguishable_exactly_at_pooled_std_is_noise():
    """The boundary is strict: |gap| must EXCEED pooled std, not merely reach it.

    Values chosen to be exactly representable in binary floating point so the
    boundary case is genuinely exact, not a rounding accident.
    """
    a = MetricValue(gross=1.0, std=3.0)
    b = MetricValue(gross=6.0, std=4.0)  # pooled = 5.0, gap = 5.0 exactly
    assert not a.distinguishable_from(b)


def test_distinguishable_one_sided_std_pools_against_zero():
    a = MetricValue(gross=0.10, std=0.02)
    deterministic = MetricValue(gross=0.15)
    assert a.distinguishable_from(deterministic)  # gap 0.05 > 0.02
    assert not a.distinguishable_from(MetricValue(gross=0.11))  # gap 0.01 < 0.02


def test_distinguishable_with_no_std_anywhere_is_false():
    """No dispersion estimate on either side = no basis to call a gap real."""
    assert not MetricValue(gross=1.0).distinguishable_from(MetricValue(gross=99.0))


def test_degraded_flags_material_cost_erosion():
    assert MetricValue(gross=0.90, net=0.30).degraded  # costs ate 2/3
    assert MetricValue(gross=0.90, net=0.80).degraded is False
    assert MetricValue(gross=0.10, net=-0.05).degraded  # sign flip is always material
    assert MetricValue(gross=0.90).degraded is False  # no net side


# ------------------------------------------------------------------ #
# PhaseResult                                                         #
# ------------------------------------------------------------------ #


def _result(**overrides) -> PhaseResult:
    base = dict(
        phase=1,
        name="Backtest engine",
        status="pass",
        gate="momentum reproduces",
        gate_passed=True,
        metrics={"sharpe": MetricValue(gross=0.61, net=0.44)},
        duration_sec=12.5,
        artifacts=[Path("reports/figures/momentum_gate.png")],
        notes=["2009 crash -34.2%"],
    )
    base.update(overrides)
    return PhaseResult(**base)


def test_phase_result_roundtrips_through_json(tmp_path):
    saved = _result().save(tmp_path)
    loaded = PhaseResult.load(saved)
    assert loaded == _result()
    assert isinstance(loaded.metrics["sharpe"], MetricValue)
    assert isinstance(loaded.artifacts[0], Path)


def test_load_all_returns_by_phase(tmp_path):
    _result(phase=0, name="Data layer").save(tmp_path)
    _result(phase=1).save(tmp_path)
    loaded = PhaseResult.load_all(tmp_path)
    assert set(loaded) == {0, 1}
    assert loaded[0].name == "Data layer"
    assert PhaseResult.load_all(tmp_path / "nowhere") == {}


def test_pass_status_requires_gate_passed():
    with pytest.raises(ValueError, match="does not pass unless its gate does"):
        _result(status="pass", gate_passed=False)


def test_invalid_phase_number_rejected():
    with pytest.raises(ValueError, match="phase must be"):
        _result(phase=11)


# ------------------------------------------------------------------ #
# net dispersion — the Session-7 contract fix                         #
# ------------------------------------------------------------------ #


def test_net_std_is_independent_of_gross_std():
    """The whole point: net dispersion is its own quantity.

    Costs scale with each seed's own turnover, so seeds can agree tightly on
    gross and disagree wildly on what survives costs. A contract that derived
    net_std from std would call this pair identical; they are not.
    """
    tight_gross_loose_net = MetricValue(gross=1.0, net=0.1, std=0.01, net_std=0.40)
    assert tight_gross_loose_net.std != tight_gross_loose_net.net_std


def test_net_distinguishable_uses_net_std_not_gross_std():
    """A gap that is real on gross can be noise on net, and vice versa."""
    a = MetricValue(gross=1.00, net=0.50, std=0.02, net_std=0.40)
    b = MetricValue(gross=0.90, net=0.20, std=0.02, net_std=0.40)

    # gross gap 0.10 vs pooled gross std 0.028 -> real
    assert a.distinguishable_from(b)
    # net gap 0.30 vs pooled net std 0.566 -> NOT real
    assert not a.net_distinguishable_from(b)


def test_net_distinguishable_can_be_real_where_gross_is_not():
    a = MetricValue(gross=1.00, net=0.90, std=0.50, net_std=0.01)
    b = MetricValue(gross=0.95, net=0.20, std=0.50, net_std=0.01)
    assert not a.distinguishable_from(b)
    assert a.net_distinguishable_from(b)


def test_net_distinguishable_strict_boundary():
    a = MetricValue(gross=0.0, net=1.0, net_std=3.0)
    b = MetricValue(gross=0.0, net=6.0, net_std=4.0)  # pooled 5.0, gap 5.0
    assert not a.net_distinguishable_from(b)


def test_net_distinguishable_without_net_raises_rather_than_falling_back():
    """Silently answering the gross question would be the dishonest failure."""
    with_net = MetricValue(gross=1.0, net=0.5, net_std=0.1)
    without = MetricValue(gross=2.0, std=0.1)
    with pytest.raises(ValueError, match="needs a net figure on both sides"):
        with_net.net_distinguishable_from(without)
    with pytest.raises(ValueError, match="needs a net figure"):
        without.net_distinguishable_from(with_net)


def test_net_distinguishable_with_no_dispersion_anywhere_is_false():
    a = MetricValue(gross=1.0, net=1.0)
    b = MetricValue(gross=1.0, net=99.0)
    assert not a.net_distinguishable_from(b)


def test_render_shows_both_error_bars_when_present():
    mv = MetricValue(gross=0.0342, net=0.0198, std=0.0061, net_std=0.0044, n_seeds=5)
    assert mv.render() == "0.0342 ± 0.0061 (net 0.0198 ± 0.0044)"


def test_render_is_unchanged_when_net_std_absent():
    """Nothing written before the fix changes appearance."""
    assert (
        MetricValue(gross=0.0342, net=0.0198, std=0.0061).render()
        == "0.0342 (net 0.0198) ± 0.0061"
    )
    assert MetricValue(gross=0.0342, std=0.0061).render() == "0.0342 ± 0.0061"


def test_phase_result_roundtrips_net_std(tmp_path):
    """Persisted results carry the new field."""
    result = _result(
        metrics={"sharpe": MetricValue(gross=0.9, net=0.4, std=0.07, net_std=0.11, n_seeds=5)}
    )
    loaded = PhaseResult.load(result.save(tmp_path))
    assert loaded.metrics["sharpe"].net_std == 0.11


def test_old_phase_json_without_net_std_still_loads(tmp_path):
    """Backwards compatible: pre-fix artifacts load with net_std=None."""
    import json

    path = tmp_path / "phase1.json"
    path.write_text(
        json.dumps(
            {
                "phase": 1,
                "name": "Backtest engine",
                "status": "pass",
                "gate": "g",
                "gate_passed": True,
                "metrics": {"sharpe": {"gross": 0.6, "net": 0.4, "std": 0.1, "n_seeds": 5}},
                "duration_sec": 1.0,
                "artifacts": [],
                "notes": [],
            }
        )
    )
    loaded = PhaseResult.load(path)
    assert loaded.metrics["sharpe"].net_std is None
    assert loaded.metrics["sharpe"].render() == "0.6000 (net 0.4000) ± 0.1000"
