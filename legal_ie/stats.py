"""Statistical helpers and visualisation utilities for legal IE analysis."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from pathlib import Path

# ── Parsing ──────────────────────────────────────────────────────────────


def iso_to_months(duration_str: str) -> int:
    """Convert an ISO 8601 duration string to a total number of months.

    Examples: ``"P2Y6M"`` → 30, ``"PT18M"`` → 18.
    Returns 0 for non-string input.
    """
    if not isinstance(duration_str, str):
        return 0

    total_months = 0
    years = re.search(r"(\d+)Y", duration_str)
    if years:
        total_months += int(years.group(1)) * 12

    months = re.search(r"(\d+)M", duration_str)
    if months:
        total_months += int(months.group(1))

    return total_months


# ── Aggregation helpers ──────────────────────────────────────────────────


def aggregate_to_set(
    df: pd.DataFrame, key_col: str, value_cols: list[str]
) -> pd.DataFrame:
    """Group by *key_col* and collect each of *value_cols* into sorted unique tuples."""
    agg_dict = {col: lambda x: tuple(sorted(set(x.dropna()))) for col in value_cols}
    return df.groupby(key_col, as_index=False).agg(agg_dict)


def explode_and_group(
    dft: pd.DataFrame,
    value_col: str,
    group_cols: tuple[str, ...] = ("ca_canonical", "OffenseType"),
) -> pd.DataFrame:
    """Explode ``OffenseType`` and *value_col*, drop nulls, group into arrays.

    Captures the repeated pattern of:

    1. ``dft.explode("OffenseType").explode(value_col)``
    2. Filter rows where *value_col* or ``OffenseType`` is null
    3. ``groupby(group_cols)[value_col].apply(np.array)``
    """
    df = dft.explode("OffenseType").explode(value_col)
    df = df[df[value_col].notnull() & df["OffenseType"].notnull()].copy()
    return df.groupby(list(group_cols))[value_col].apply(np.array).reset_index()


def weighted_mean_by_group(
    df: pd.DataFrame,
    value_col: str,
    weight_col: str,
    group_cols: list[str] | tuple[str, ...],
) -> pd.Series:
    """Weighted mean of *value_col* by *weight_col* within *group_cols*."""
    cols = list(group_cols) + [value_col, weight_col]
    return df[cols].groupby(list(group_cols)).apply(
        lambda g: np.dot(g[value_col], g[weight_col]) / g[weight_col].sum()
    )


def replace_small_counts(
    series: pd.Series, small_value: int = 2, nc_value: int = 5
) -> pd.Series:
    """Replace ``'<5'`` → *small_value* and ``'nc'`` → *nc_value*, then cast to int."""
    return series.replace("<5", small_value).replace("nc", nc_value).astype(int)


# ── Infra-mapping display ───────────────────────────────────────────────


def dump_infra_mapping(df_crime_types_mapping: pd.DataFrame) -> pd.DataFrame:
    """Decode notation symbols in crime-type mapping; return a readable DataFrame."""

    def _decode_notation(row: pd.Series) -> str:
        n, left, right = row[["Notation", "Ontology_Class", "Ministry_Class"]].values
        notation_map = {
            r"$\supset$": f"({right}) is part of ({left})",
            r"$\subset$": f"({left}) is part of ({right})",
            r"$\cap$": f"({left}) and ({right}) have non-empty intersection",
            r"$\approx$": f"({left}) and ({right}) are the same",
        }
        return notation_map.get(n, "error")

    df_tmp = (
        df_crime_types_mapping.loc[
            df_crime_types_mapping.Ministry_Class.notnull(),
            ["Ontology_Class", "Ministry_Class", "Notation"],
        ]
        .sort_values(["Ontology_Class", "Notation", "Ministry_Class"])
        .reset_index(drop=True)
    )
    df_tmp["Explanation"] = df_tmp.apply(_decode_notation, axis=1)
    return df_tmp


# ── Age bin helpers ──────────────────────────────────────────────────────


def compute_age_bins(
    columns_nb: list[str],
) -> tuple[list[str], np.ndarray, list[float]]:
    """Derive age-bin edges and centres from ``NB_*`` column names.

    Returns ``(columns_ages, bins, age_group_centers)``.
    """
    columns_ages = [c for c in columns_nb if c.split("_")[1][0].isnumeric()]
    bins = np.array([float(c.split("_")[1][:2]) for c in columns_ages] + [120.0])
    age_group_centers = [float(np.mean(pair)) for pair in zip(bins, bins[1:])]
    return columns_ages, bins, age_group_centers


# ── Visualisation ────────────────────────────────────────────────────────


def plot_parity(
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    output_path: Path | None = None,
    ax=None,
):
    """Scatter plot with a 45° parity line comparing two columns."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    ax.scatter(df[col_x], df[col_y], alpha=0.6, edgecolors="k", linewidths=0.5)

    lo = min(df[col_x].min(), df[col_y].min())
    hi = max(df[col_x].max(), df[col_y].max())
    margin = (hi - lo) * 0.05 or 1.0
    ax.plot(
        [lo - margin, hi + margin],
        [lo - margin, hi + margin],
        ls="--",
        color="grey",
        linewidth=1,
    )

    ax.set_xlabel(xlabel or col_x)
    ax.set_ylabel(ylabel or col_y)
    ax.set_title(title or f"{col_x} vs {col_y}")
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    return ax


def plot_distribution_comparison(
    labels,
    values_a,
    values_b,
    *,
    name_a: str = "Ontology",
    name_b: str = "Official Stats",
    title: str = "Distribution Comparison",
    ylabel: str = "Proportion",
    output_path: Path | None = None,
    ax=None,
    rotate_labels: int = 45,
):
    """Side-by-side bar chart comparing two distributions over shared categories."""
    import matplotlib.pyplot as plt

    x = np.arange(len(labels))
    width = 0.35

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 6))
    else:
        fig = ax.figure

    ax.bar(x - width / 2, values_a, width, label=name_a, alpha=0.8)
    ax.bar(x + width / 2, values_b, width, label=name_b, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate_labels, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    if output_path:
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    return ax
