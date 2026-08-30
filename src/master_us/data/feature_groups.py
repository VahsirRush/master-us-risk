"""The feature group registry — spec section 4.3's table, made executable.

Defined before `features.py` exists, deliberately. `RobustZScoreNorm` emits one
missingness indicator per group, and without a registry its only safe default is
one indicator per feature — which at the target of ~150 features means 150 extra
columns carrying almost no information, since features in a group go missing
together (no price history means every return window is absent at once, not
`ret_5d` alone).

Naming convention, fixed here so `features.py` has something to build against:

    ret_<window>          returns          ret_1d, ret_20d
    vol_<detail>          volatility       vol_20d, vol_parkinson_20d
    volume_<detail>       volume           volume_ratio_20d, volume_amihud_60d
    ps_<detail>           price structure  ps_hl_range, ps_overnight_gap
    tech_<detail>         technical        tech_rsi_14, tech_beta_60d
    xs_<base feature>     cross-sectional  xs_ret_20d, xs_tech_rsi_14

Prefix matching is longest-wins, which is load-bearing: `vol_` is a prefix of
`volume_`, so a naive first-match would file every volume feature under
volatility. `test_feature_groups.py` pins that case.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from enum import Enum


class FeatureGroup(Enum):
    """The six groups of spec section 4.3, plus a catch-all.

    `OTHER` exists so that an unrecognized name is absorbed visibly rather than
    crashing a pipeline mid-run. It is not a licence to ignore it — pass
    `strict=True` to `classify_features` (and `strict_groups=True` to
    `RobustZScoreNorm`) on any path that builds a real panel, and OTHER becomes
    an error.
    """

    RETURNS = "returns"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    PRICE_STRUCTURE = "price_structure"
    TECHNICAL = "technical"
    CROSS_SECTIONAL_RANK = "cross_sectional_rank"
    OTHER = "other"

    def __str__(self) -> str:
        return self.value


# Ordered longest-first so that `volume_` wins over `vol_`. Do not reorder into
# something prettier; `classify_feature` depends on this ordering.
GROUP_PREFIXES: tuple[tuple[str, FeatureGroup], ...] = tuple(
    sorted(
        [
            ("ret_", FeatureGroup.RETURNS),
            ("vol_", FeatureGroup.VOLATILITY),
            ("volume_", FeatureGroup.VOLUME),
            ("ps_", FeatureGroup.PRICE_STRUCTURE),
            ("tech_", FeatureGroup.TECHNICAL),
            ("xs_", FeatureGroup.CROSS_SECTIONAL_RANK),
        ],
        key=lambda kv: len(kv[0]),
        reverse=True,
    )
)

# What spec section 4.3 says belongs in each group. Carried in code so the
# feature bank has a checklist next session, and so a reviewer can see that the
# registry matches the spec table without opening the spec.
GROUP_CONTENTS: Mapping[FeatureGroup, tuple[str, ...]] = {
    FeatureGroup.RETURNS: ("close-to-close over 1, 2, 3, 5, 10, 20, 30, 60d",),
    FeatureGroup.VOLATILITY: (
        "realized std over 1, 2, 3, 5, 10, 20, 30, 60d",
        "Parkinson",
        "Garman-Klass",
    ),
    FeatureGroup.VOLUME: (
        "volume ratio vs. 20/60d mean",
        "dollar-volume z-score",
        "Amihud illiquidity",
    ),
    FeatureGroup.PRICE_STRUCTURE: (
        "(high - low) / close",
        "close position within day range",
        "VWAP deviation",
        "overnight gap",
    ),
    FeatureGroup.TECHNICAL: (
        "RSI(14)",
        "MACD(12, 26, 9)",
        "Bollinger %B(20, 2)",
        "rolling beta(60) to S&P",
        "rolling corr(60) to S&P",
    ),
    FeatureGroup.CROSS_SECTIONAL_RANK: ("per-date rank transform of every feature above",),
    FeatureGroup.OTHER: ("unrecognized — a naming bug, or a feature added without a group",),
}


class UnknownFeatureGroupError(ValueError):
    """Raised when a feature name matches no group and strict classification is on."""


def classify_feature(name: str) -> FeatureGroup:
    """Map one feature name to its group by longest matching prefix.

    Cross-sectional ranks are their own group rather than inheriting the group
    of the feature they rank. That follows the spec table, and it is worth
    knowing that the resulting indicator column is largely redundant: rank(NaN)
    is NaN, so the `cross_sectional_rank` mask is close to the OR of every other
    group's mask. Kept anyway — the spec is explicit, and the redundancy costs
    one column, not one per feature.
    """
    for prefix, group in GROUP_PREFIXES:
        if name.startswith(prefix):
            return group
    return FeatureGroup.OTHER


def classify_features(
    names: Iterable[str],
    strict: bool = False,
) -> dict[FeatureGroup, list[str]]:
    """Group feature names, preserving input order within each group.

    Only groups that actually have members are returned — an empty group would
    emit an all-False indicator column, which is a wasted column and a
    misleading one.

    Parameters
    ----------
    strict:
        Raise `UnknownFeatureGroupError` if any name falls to `OTHER`. Use this
        on every path that builds a real panel: a feature silently filed under
        OTHER has its missingness pooled with unrelated features, which quietly
        degrades the indicator without failing anything.
    """
    grouped: defaultdict[FeatureGroup, list[str]] = defaultdict(list)
    for name in names:
        grouped[classify_feature(name)].append(name)

    unknown = grouped.get(FeatureGroup.OTHER, [])
    if strict and unknown:
        shown = ", ".join(unknown[:8])
        more = f", ... (+{len(unknown) - 8} more)" if len(unknown) > 8 else ""
        raise UnknownFeatureGroupError(
            f"{len(unknown)} feature name(s) match no group prefix: {shown}{more}. "
            f"Expected one of {sorted(p for p, _ in GROUP_PREFIXES)}. "
            "Rename the feature, or add a prefix to GROUP_PREFIXES if it is a new group."
        )

    return {group: grouped[group] for group in FeatureGroup if group in grouped}


def group_names(names: Iterable[str], strict: bool = False) -> dict[str, list[str]]:
    """`classify_features` keyed by the group's string value.

    The form `RobustZScoreNorm` consumes, since indicator columns are named
    after the group and a `FeatureGroup` key would stringify as `FeatureGroup.RETURNS`.
    """
    return {str(group): members for group, members in classify_features(names, strict).items()}


def coverage_table(names: Iterable[str]) -> list[tuple[str, int]]:
    """(group, count) for every group present. Goes in NOTES.md at Phase 0."""
    return [(str(g), len(m)) for g, m in classify_features(names).items()]
