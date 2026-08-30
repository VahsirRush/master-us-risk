"""Tests for the feature group registry — spec section 4.3.

The registry is a naming convention with consequences: it decides which features
share a missingness indicator. A misfiled feature does not crash anything, it
just pools its NaN pattern with unrelated features, so the tests here are mostly
about the ways a prefix scheme quietly goes wrong.
"""

from __future__ import annotations

import pytest

from master_us.data.feature_groups import (
    GROUP_CONTENTS,
    GROUP_PREFIXES,
    FeatureGroup,
    UnknownFeatureGroupError,
    classify_feature,
    classify_features,
    coverage_table,
    group_names,
)


def test_volume_does_not_get_filed_under_volatility():
    """`vol_` is a prefix of `volume_`. First-match ordering gets this wrong.

    This is the single most likely defect in a prefix scheme, and it fails
    silently — every volume feature would land in volatility and the panel would
    still build.
    """
    assert classify_feature("vol_20d") is FeatureGroup.VOLATILITY
    assert classify_feature("vol_parkinson_20d") is FeatureGroup.VOLATILITY
    assert classify_feature("volume_ratio_20d") is FeatureGroup.VOLUME
    assert classify_feature("volume_amihud_60d") is FeatureGroup.VOLUME


def test_prefixes_are_ordered_longest_first():
    """The ordering is load-bearing, so assert it rather than trusting the sort."""
    lengths = [len(prefix) for prefix, _ in GROUP_PREFIXES]
    assert lengths == sorted(lengths, reverse=True)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ret_1d", FeatureGroup.RETURNS),
        ("ret_60d", FeatureGroup.RETURNS),
        ("vol_gk_20d", FeatureGroup.VOLATILITY),
        ("volume_dollar_zscore_20d", FeatureGroup.VOLUME),
        ("ps_overnight_gap", FeatureGroup.PRICE_STRUCTURE),
        ("ps_vwap_deviation", FeatureGroup.PRICE_STRUCTURE),
        ("tech_rsi_14", FeatureGroup.TECHNICAL),
        ("tech_macd_12_26_9", FeatureGroup.TECHNICAL),
        ("tech_beta_60d", FeatureGroup.TECHNICAL),
        ("xs_ret_20d", FeatureGroup.CROSS_SECTIONAL_RANK),
        ("xs_tech_rsi_14", FeatureGroup.CROSS_SECTIONAL_RANK),
        ("f000", FeatureGroup.OTHER),
        ("mystery", FeatureGroup.OTHER),
        ("", FeatureGroup.OTHER),
    ],
)
def test_classification_covers_the_spec_table(name, expected):
    assert classify_feature(name) is expected


def test_cross_sectional_rank_does_not_inherit_the_base_group():
    """`xs_ret_20d` is a rank, not a return. Spec section 4.3 lists ranks separately."""
    assert classify_feature("ret_20d") is FeatureGroup.RETURNS
    assert classify_feature("xs_ret_20d") is FeatureGroup.CROSS_SECTIONAL_RANK


def test_every_spec_group_has_a_prefix_and_documented_contents():
    """A group nothing can be classified into would emit an all-False indicator."""
    prefixed = {group for _, group in GROUP_PREFIXES}
    assert prefixed == set(FeatureGroup) - {FeatureGroup.OTHER}
    assert set(GROUP_CONTENTS) == set(FeatureGroup)


def test_grouping_preserves_order_and_omits_empty_groups():
    names = ["ret_5d", "tech_rsi_14", "ret_1d", "tech_beta_60d"]
    grouped = classify_features(names)

    assert list(grouped) == [FeatureGroup.RETURNS, FeatureGroup.TECHNICAL]
    assert grouped[FeatureGroup.RETURNS] == ["ret_5d", "ret_1d"]
    assert grouped[FeatureGroup.TECHNICAL] == ["tech_rsi_14", "tech_beta_60d"]
    assert FeatureGroup.VOLUME not in grouped


def test_groups_are_returned_in_spec_table_order():
    """Indicator column order must not depend on the order features arrive in."""
    forward = classify_features(["ret_1d", "vol_20d", "volume_ratio_20d", "xs_ret_1d"])
    backward = classify_features(["xs_ret_1d", "volume_ratio_20d", "vol_20d", "ret_1d"])
    assert list(forward) == list(backward)
    assert list(forward) == [
        FeatureGroup.RETURNS,
        FeatureGroup.VOLATILITY,
        FeatureGroup.VOLUME,
        FeatureGroup.CROSS_SECTIONAL_RANK,
    ]


def test_strict_mode_rejects_unclassifiable_names():
    names = ["ret_1d", "wat", "alsowat"]

    lenient = classify_features(names, strict=False)
    assert lenient[FeatureGroup.OTHER] == ["wat", "alsowat"]

    with pytest.raises(UnknownFeatureGroupError, match="match no group prefix"):
        classify_features(names, strict=True)


def test_strict_mode_error_names_the_offenders():
    with pytest.raises(UnknownFeatureGroupError) as exc:
        classify_features(["ret_1d", "typo_ret_5d"], strict=True)
    assert "typo_ret_5d" in str(exc.value)
    assert "ret_1d" not in str(exc.value).split("prefix:")[1].split(".")[0]


def test_strict_mode_passes_when_everything_classifies():
    names = ["ret_1d", "vol_20d", "volume_ratio_20d", "ps_hl_range", "tech_rsi_14", "xs_ret_1d"]
    assert FeatureGroup.OTHER not in classify_features(names, strict=True)


def test_group_names_keys_are_plain_strings():
    """`RobustZScoreNorm` names indicator columns from these keys.

    A `FeatureGroup` key would stringify as `FeatureGroup.RETURNS` and produce
    a column called `FeatureGroup.RETURNS__isnan`.
    """
    keyed = group_names(["ret_1d", "tech_rsi_14"])
    assert list(keyed) == ["returns", "technical"]
    assert all(isinstance(k, str) and not k.startswith("FeatureGroup") for k in keyed)


def test_coverage_table_counts_members():
    table = coverage_table(["ret_1d", "ret_5d", "ret_20d", "tech_rsi_14", "nope"])
    assert table == [("returns", 3), ("technical", 1), ("other", 1)]
