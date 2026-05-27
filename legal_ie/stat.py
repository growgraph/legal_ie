"""Statistical helpers and visualisation utilities for legal IE analysis."""

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
    group_cols: tuple[str, ...] = ("ca_canonical",),
    multi_groups: tuple[str, ...] = tuple(),
) -> pd.DataFrame:
    """Explode ``OffenseType`` and *value_col*, drop nulls, group into arrays.

    Captures the repeated pattern of:

    1. ``dft.explode("OffenseType").explode(value_col)``
    2. Filter rows where *value_col* or ``OffenseType`` is null
    3. ``groupby(group_cols)[value_col].apply(np.array)``
    """
    group_cols = list(set(list(set(group_cols)) + list(set(multi_groups))))
    df = dft.copy()

    for mg in multi_groups:
        df = df.explode(mg)
    df = df.explode(value_col)
    df = df[df[value_col].notnull()].copy()
    return df.groupby(list(group_cols))[value_col].apply(np.array).reset_index()


def weighted_mean_by_group(
    df: pd.DataFrame,
    value_col: str,
    weight_col: str,
    group_cols: list[str] | tuple[str, ...],
    discard_nulls: bool = True,
) -> pd.Series:
    """Weighted mean of *value_col* by *weight_col* within *group_cols*."""
    cols = list(group_cols) + [value_col, weight_col]
    if discard_nulls:
        df = df.loc[~df[value_col].isnull()].copy()

    return (
        df[cols]
        .groupby(list(group_cols))
        .apply(lambda g: np.dot(g[value_col], g[weight_col]) / g[weight_col].sum())
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
        n, left, right = row[
            [
                "Notation",
                "InfractionClassOntologyClassification",
                "InfractionClassMinistryClassification",
            ]
        ].values
        notation_map = {
            r"$\supset$": f"({right}) is part of ({left})",
            r"$\subset$": f"({left}) is part of ({right})",
            r"$\cap$": f"({left}) and ({right}) have non-empty intersection",
            r"$\approx$": f"({left}) and ({right}) are the same",
        }
        return notation_map.get(n, "error")

    df_tmp = (
        df_crime_types_mapping.loc[
            df_crime_types_mapping.InfractionClassMinistryClassification.notnull(),
            [
                "InfractionClassOntologyClassification",
                "InfractionClassMinistryClassification",
                "Notation",
            ],
        ]
        .sort_values(
            [
                "InfractionClassOntologyClassification",
                "Notation",
                "InfractionClassMinistryClassification",
            ]
        )
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


def compare_counts(
    df_counts: pd.Series, df_author_gb: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare case counts by geography and by crime type.

    Returns ``(df_counts_ca, df_counts_infra)``.
    """
    # By geography (ca_canonical)
    df_ca = pd.merge(
        df_author_gb["NB_AUT"].groupby(level=0).sum().rename("cnt_ca_stat"),
        df_counts.groupby(level=0).sum().rename("cnt_ca"),
        left_index=True,
        right_index=True,
    )
    df_ca["dist_ca_stat"] = df_ca["cnt_ca_stat"] / df_ca["cnt_ca_stat"].sum()
    df_ca["dist_ca"] = df_ca["cnt_ca"] / df_ca["cnt_ca"].sum()

    # By crime type (OffenseType / InfractionClassOntologyClassification)
    df_infra = pd.merge(
        df_author_gb["NB_AUT"].groupby(level=1).sum().rename("cnt_infra_stat"),
        df_counts.groupby(level=1).sum().rename("cnt_infra"),
        left_index=True,
        right_index=True,
    )
    df_infra["dist_infra_stat"] = (
        df_infra["cnt_infra_stat"] / df_infra["cnt_infra_stat"].sum()
    )
    df_infra["dist_infra"] = df_infra["cnt_infra"] / df_infra["cnt_infra"].sum()

    return df_ca, df_infra


def compare_metric(
    df_onto: pd.DataFrame,
    df_stat: pd.Series | pd.DataFrame,
    value_col: str,
    mean_fn: callable,
    onto_mean_name: str = "mean_onto",
    stat_name: str | None = None,
    offense_type_col: str = "OffenseType",
    infraction_col: str = "InfractionClassOntologyClassification",
) -> pd.DataFrame:
    """Compare an ontology-derived metric with official statistics.

    Automatically detects the grouping level from *df_onto*:

    * If *offense_type_col* is present the merge uses
      ``(ca_canonical, <infraction_col>)``.
    * Otherwise the merge uses ``ca_canonical`` only, preserving data
      points that lack an infraction-class label.

    Parameters
    ----------
    df_onto : DataFrame
        Ontology-side aggregation (output of :func:`explode_and_group`).
    df_stat : Series or DataFrame
        Official-statistics reference, indexed by the same group columns.
    value_col : str
        Column in *df_onto* containing the raw arrays to aggregate.
    mean_fn : callable
        Applied element-wise to *value_col* to compute the mean
        (e.g. ``np.mean`` for ages, a custom function for durations).
    onto_mean_name : str
        Name for the computed mean column in the result.
    stat_name : str or None
        If *df_stat* is a Series, rename it to *stat_name* before merging.
    offense_type_col : str
        Column in *df_onto* to detect and rename (default ``"OffenseType"``).
    infraction_col : str
        Target column name after renaming (default
        ``"InfractionClassOntologyClassification"``).
    """
    df = df_onto.copy()
    df[onto_mean_name] = df[value_col].apply(mean_fn)

    group_cols = ["ca_canonical"]
    if offense_type_col in df.columns:
        df = df.rename(columns={offense_type_col: infraction_col})
        group_cols.append(infraction_col)

    onto_indexed = df.set_index(group_cols)[[onto_mean_name]]

    if isinstance(df_stat, pd.Series):
        stat_frame = (
            df_stat.rename(stat_name).to_frame() if stat_name else df_stat.to_frame()
        )
    else:
        stat_frame = df_stat

    return pd.merge(
        stat_frame,
        onto_indexed,
        left_index=True,
        right_index=True,
    )


def derive_official_age(
    df_author: pd.DataFrame,
    columns_nb: list[str],
    group_cols: list[str] | tuple[str, ...] = (
        "ca_canonical",
        "InfractionClassOntologyClassification",
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute weighted-mean age from official author statistics.

    Returns ``(df_age_stat, df_author_gb)`` where *df_author_gb* is the
    grouped sum used also for count comparisons.
    """
    group_cols = list(group_cols)
    df_gb = df_author.groupby(group_cols)[columns_nb].sum()
    columns_ages, _, centers = compute_age_bins(columns_nb)

    df_age_stat = (
        df_gb.apply(
            lambda x: np.dot(x[columns_ages], centers) / x[columns_ages].sum(),
            axis=1,
        )
        .rename("mean_age_stat")
        .to_frame()
    )
    return df_age_stat, df_gb


def derive_ontology_counts(
    dft: pd.DataFrame, offense_col: str = "OffenseType"
) -> pd.Series:
    """Count individuals per ``(ca_canonical, offense_col)`` from ontology data."""
    df = dft.explode(offense_col)
    df = df.loc[df[offense_col].notnull()].copy()
    return df.groupby(["ca_canonical", offense_col])["Age"].apply(
        lambda x: sum(len(xx) if xx else 1 for xx in x)
    )


def prepare_official_stats(
    df_stat_author: pd.DataFrame,
    df_stat_conda: pd.DataFrame,
    df_ca_tj: pd.DataFrame,
    df_crime_mapping_onto: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Join official statistics with bridge and crime-type mapping.

    Returns ``(df_author_prep, df_conda_prep, columns_nb)``.
    """
    bridge = df_ca_tj[["tj_canonical", "tj_stat", "ca_canonical"]].drop_duplicates(
        ["tj_canonical", "tj_stat"]
    )

    df_author = pd.merge(
        df_stat_author, bridge, left_on="LIB_JUR", right_on="tj_stat", how="left"
    )
    df_conda = pd.merge(
        df_stat_conda, bridge, left_on="LIB_JUR", right_on="tj_stat", how="left"
    )

    mapping = df_crime_mapping_onto.drop("Notation", axis=1)
    df_author = pd.merge(
        df_author,
        mapping,
        left_on="GR_INF",
        right_on="InfractionClassMinistryClassification",
        how="inner",
    )
    df_conda = pd.merge(
        df_conda,
        mapping,
        left_on="GR_INF",
        right_on="InfractionClassMinistryClassification",
        how="inner",
    )

    # Clean small / missing counts
    columns_nb = [c for c in df_stat_author.columns if c.startswith("NB")]
    for c in columns_nb:
        df_author[c] = replace_small_counts(df_author[c])
    df_conda["NB_CONDA"] = replace_small_counts(df_conda["NB_CONDA"])

    return df_author, df_conda, columns_nb


def load_all_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all input CSV files from *data_dir*."""
    df_data = pd.read_csv(data_dir / "case-age-fine-duration-offence-decision.csv")
    df_data["FineAmount"] = df_data["FineAmount"].apply(pd.to_numeric, errors="coerce")

    df_crime_types_mapping = pd.read_csv(data_dir / "crime_types_cc_stat.csv", sep=";")
    df_mentions = pd.read_csv(data_dir / "mentions.csv")
    df_ca_tj = pd.read_csv(data_dir / "bridge.csv")

    df_stat_author = pd.read_csv(data_dir / "AUTEUR_s.csv", sep=";")
    df_stat_conda = pd.read_csv(data_dir / "CONDA_s.csv", sep=";")
    df_stat_author = df_stat_author.drop(df_stat_author.columns[0], axis=1)
    df_stat_conda = df_stat_conda.drop(df_stat_conda.columns[0], axis=1)

    return dict(
        data=df_data,
        crime_types_mapping=df_crime_types_mapping,
        mentions=df_mentions,
        ca_tj=df_ca_tj,
        stat_author=df_stat_author,
        stat_conda=df_stat_conda,
    )


def prepare_merged_data(
    df_data: pd.DataFrame, df_mentions: pd.DataFrame, df_ca_tj: pd.DataFrame
) -> pd.DataFrame:
    """Merge case data with court mentions and CA–TJ bridge → ``dft``."""
    df_data["AppealId"] = df_data["g"].apply(lambda x: x.split("/")[-1])
    df_data_set = aggregate_to_set(
        df_data, "AppealId", ["Age", "PunishmentDuration", "FineAmount", "OffenseType"]
    )

    df_mentions["AppealId"] = df_mentions["filename"].apply(
        lambda x: x.split("_")[1].split("°")[1]
    )
    df_mentions = (
        df_mentions.sort_values(["AppealId", "mention"])
        .groupby("AppealId", as_index=False)
        .first()
    )
    df_mentions["mention"] = df_mentions["mention"].str.lower()

    dft = pd.merge(df_data_set, df_mentions, on="AppealId")
    dft = pd.merge(
        dft,
        df_ca_tj[["ca_mention", "ca_canonical"]].drop_duplicates(
            ["ca_mention", "ca_canonical"]
        ),
        left_on="mention",
        right_on="ca_mention",
    )
    return dft


def add_ministry_classification(
    dft: pd.DataFrame,
    df_crime_mapping: pd.DataFrame,
    source_col: str = "OffenseType",
    target_col: str = "MinistryClassType",
) -> pd.DataFrame:
    """Map ontology *source_col* tuples to ministry-classification tuples.

    Each element in the *source_col* tuple (an ontology classification) is
    mapped through *df_crime_mapping* to its corresponding
    ``InfractionClassMinistryClassification`` value(s).  The results are
    collected into a sorted tuple stored in *target_col*.
    """
    mc_lookup = (
        df_crime_mapping[
            [
                "InfractionClassOntologyClassification",
                "InfractionClassMinistryClassification",
            ]
        ]
        .drop_duplicates()
        .groupby("InfractionClassOntologyClassification")[
            "InfractionClassMinistryClassification"
        ]
        .apply(set)
        .to_dict()
    )

    def _map_row(offense_types):
        result = set()
        for ot in offense_types if offense_types else ():
            if pd.notna(ot) and ot in mc_lookup:
                result.update(mc_lookup[ot])
        return tuple(sorted(result)) if result else tuple()

    dft = dft.copy()
    dft[target_col] = dft[source_col].apply(_map_row)
    return dft


def prepare_crime_type_filters(
    df_crime_types_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the mapping subsets for ontology-side and stat-side joins.

    ``df_gb_onto`` excludes ``$\\subset$`` rows (for joining official stats
    onto ontology classes).  ``df_gb_stat`` excludes ``$\\supset$`` rows.
    """
    df_notnull = df_crime_types_mapping.loc[
        df_crime_types_mapping.InfractionClassMinistryClassification.notnull(),
        df_crime_types_mapping.columns[:3],
    ].copy()
    df_gb_onto = df_notnull[df_notnull["Notation"] != r"$\subset$"].copy()
    df_gb_stat = df_notnull[df_notnull["Notation"] != r"$\supset$"].copy()
    return df_gb_onto, df_gb_stat
