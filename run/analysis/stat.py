"""CLI for comparing ontology-extracted statistics with official crime statistics."""

import click
import numpy as np
import pandas as pd
from pathlib import Path

from legal_ie.stats import (
    iso_to_months,
    aggregate_to_set,
    explode_and_group,
    weighted_mean_by_group,
    replace_small_counts,
    compute_age_bins,
    plot_parity,
    plot_distribution_comparison,
)


# ── Data loading ─────────────────────────────────────────────────────────


def load_all_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all input CSV files from *data_dir*."""
    df_data = pd.read_csv(data_dir / "case-age-fine-duration-offence-decision.csv")
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


# ── Ontology data preparation ───────────────────────────────────────────


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


# ── Crime-type mapping filters ──────────────────────────────────────────


def prepare_crime_type_filters(
    df_crime_types_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the mapping subsets for ontology-side and stat-side joins.

    ``df_gb_onto`` excludes ``$\\subset$`` rows (for joining official stats
    onto ontology classes).  ``df_gb_stat`` excludes ``$\\supset$`` rows.
    """
    df_notnull = df_crime_types_mapping.loc[
        df_crime_types_mapping.Ministry_Class.notnull(),
        df_crime_types_mapping.columns[:3],
    ].copy()
    df_gb_onto = df_notnull[df_notnull["Notation"] != r"$\subset$"].copy()
    df_gb_stat = df_notnull[df_notnull["Notation"] != r"$\supset$"].copy()
    return df_gb_onto, df_gb_stat


# ── Official stats preparation ──────────────────────────────────────────


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
        df_author, mapping, left_on="GR_INF", right_on="Ministry_Class", how="inner"
    )
    df_conda = pd.merge(
        df_conda, mapping, left_on="GR_INF", right_on="Ministry_Class", how="inner"
    )

    # Clean small / missing counts
    columns_nb = [c for c in df_stat_author.columns if c.startswith("NB")]
    for c in columns_nb:
        df_author[c] = replace_small_counts(df_author[c])
    df_conda["NB_CONDA"] = replace_small_counts(df_conda["NB_CONDA"])

    return df_author, df_conda, columns_nb


# ── Ontology-side stat derivation ────────────────────────────────────────


def derive_ontology_counts(dft: pd.DataFrame) -> pd.Series:
    """Count individuals per ``(ca_canonical, OffenseType)`` from ontology data."""
    df = dft.explode("OffenseType")
    df = df.loc[df["OffenseType"].notnull()].copy()
    return df.groupby(["ca_canonical", "OffenseType"])["Age"].apply(
        lambda x: sum(len(xx) if xx else 1 for xx in x)
    )


# ── Official-side stat derivation ───────────────────────────────────────


def derive_official_age(
    df_author: pd.DataFrame, columns_nb: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute weighted-mean age from official author statistics.

    Returns ``(df_age_stat, df_author_gb)`` where *df_author_gb* is the
    grouped sum used also for count comparisons.
    """
    df_gb = df_author.groupby(["ca_canonical", "Ontology_Class"])[columns_nb].sum()
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


# ── Comparison merges ────────────────────────────────────────────────────


def compare_ages(df_age_onto: pd.DataFrame, df_age_stat: pd.DataFrame) -> pd.DataFrame:
    """Merge ontology mean-age with official mean-age."""
    df = df_age_onto.rename(columns={"OffenseType": "Ontology_Class"})
    df["mean_age"] = df["Age"].apply(np.mean)
    return pd.merge(
        df[["ca_canonical", "Ontology_Class", "mean_age"]],
        df_age_stat,
        left_on=["ca_canonical", "Ontology_Class"],
        right_index=True,
    )


def compare_durations(
    df_dur_onto: pd.DataFrame, df_dur_stat: pd.Series
) -> pd.DataFrame:
    """Merge ontology mean-duration with official mean-duration."""
    df_dur_onto = df_dur_onto.copy()
    df_dur_onto["mean_dur_years"] = df_dur_onto["PunishmentDuration"].apply(
        lambda x: np.mean([iso_to_months(d) for d in x]) / 12
    )
    return pd.merge(
        df_dur_stat.rename("dur_stat").to_frame(),
        df_dur_onto.rename(columns={"OffenseType": "Ontology_Class"})
        .set_index(["ca_canonical", "Ontology_Class"])
        .mean_dur_years.to_frame(),
        left_index=True,
        right_index=True,
    )


def compare_fines(df_fine_onto: pd.DataFrame, df_fine_stat: pd.Series) -> pd.DataFrame:
    """Merge ontology mean-fine with official mean-fine."""
    df_fine_onto = df_fine_onto.copy()
    df_fine_onto["mean_fine"] = df_fine_onto["FineAmount"].apply(
        lambda x: np.mean([xx for xx in x if not isinstance(xx, str)])
    )
    return pd.merge(
        df_fine_stat.rename("fine_stat").to_frame(),
        df_fine_onto.rename(columns={"OffenseType": "Ontology_Class"})
        .set_index(["ca_canonical", "Ontology_Class"])
        .mean_fine.to_frame(),
        left_index=True,
        right_index=True,
    )


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

    # By crime type (OffenseType / Ontology_Class)
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


# ── Visualisation ────────────────────────────────────────────────────────


def generate_plots(
    df_age_final: pd.DataFrame,
    df_dur_final: pd.DataFrame,
    df_fine_final: pd.DataFrame,
    df_counts_ca: pd.DataFrame,
    df_counts_infra: pd.DataFrame,
    plots_dir: Path,
) -> None:
    """Save all comparison plots to *plots_dir*."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Age: parity plot
    plot_parity(
        df_age_final,
        "mean_age",
        "mean_age_stat",
        xlabel="Mean age (ontology)",
        ylabel="Mean age (official stats)",
        title="Age: ontology vs official statistics",
        output_path=plots_dir / "age_comparison.png",
    )

    # Duration: parity plot
    plot_parity(
        df_dur_final.reset_index(),
        "mean_dur_years",
        "dur_stat",
        xlabel="Mean duration, years (ontology)",
        ylabel="Mean duration, years (official stats)",
        title="Imprisonment duration: ontology vs official statistics",
        output_path=plots_dir / "duration_comparison.png",
    )

    # Fine: parity plot
    plot_parity(
        df_fine_final.reset_index(),
        "mean_fine",
        "fine_stat",
        xlabel="Mean fine amount (ontology)",
        ylabel="Mean fine amount (official stats)",
        title="Fine amount: ontology vs official statistics",
        output_path=plots_dir / "fine_comparison.png",
    )

    # Geographic distribution
    plot_distribution_comparison(
        labels=df_counts_ca.index.tolist(),
        values_a=df_counts_ca["dist_ca"].values,
        values_b=df_counts_ca["dist_ca_stat"].values,
        name_a="Ontology",
        name_b="Official Stats",
        title="Geographic distribution of cases",
        output_path=plots_dir / "geography_distribution.png",
    )

    # Crime-type distribution
    plot_distribution_comparison(
        labels=df_counts_infra.index.tolist(),
        values_a=df_counts_infra["dist_infra"].values,
        values_b=df_counts_infra["dist_infra_stat"].values,
        name_a="Ontology",
        name_b="Official Stats",
        title="Crime-type distribution",
        output_path=plots_dir / "crime_type_distribution.png",
    )


# ── CLI ──────────────────────────────────────────────────────────────────


@click.command()
@click.option(
    "--data-dir",
    default="~/data/legal/working",
    help="Directory containing the CSV files",
    type=click.Path(),
)
@click.option(
    "--output",
    default="merged.csv",
    help="Path to output the merged CSV",
    type=click.Path(),
)
@click.option(
    "--plots-dir",
    default="~/data/legal/plots",
    help="Directory to save comparison plots",
    type=click.Path(),
)
def merge_csv(data_dir, output, plots_dir):
    """Compare ontology-extracted statistics with official crime statistics."""
    data_dir = Path(data_dir).expanduser()
    plots_dir = Path(plots_dir).expanduser()

    # ── Load ─────────────────────────────────────────────────────────────
    data = load_all_data(data_dir)

    # ── Prepare merged ontology data ─────────────────────────────────────
    dft = prepare_merged_data(data["data"], data["mentions"], data["ca_tj"])

    # ── Ontology-side stats ──────────────────────────────────────────────
    df_age_onto = explode_and_group(dft, "Age")
    df_fine_onto = explode_and_group(dft, "FineAmount")
    df_dur_onto = explode_and_group(dft, "PunishmentDuration")
    df_counts = derive_ontology_counts(dft)

    # ── Official-side stats ──────────────────────────────────────────────
    df_crime_onto, _ = prepare_crime_type_filters(data["crime_types_mapping"])
    df_author, df_conda, columns_nb = prepare_official_stats(
        data["stat_author"], data["stat_conda"], data["ca_tj"], df_crime_onto
    )

    df_dur_stat = weighted_mean_by_group(
        df_conda, "QMF_EMP_F", "NB_CONDA", ["ca_canonical", "Ontology_Class"]
    )
    df_fine_stat = weighted_mean_by_group(
        df_conda, "MMF_AME_F", "NB_CONDA", ["ca_canonical", "Ontology_Class"]
    )
    df_age_stat, df_author_gb = derive_official_age(df_author, columns_nb)

    # ── Comparisons ──────────────────────────────────────────────────────
    df_age_final = compare_ages(df_age_onto, df_age_stat)
    df_dur_final = compare_durations(df_dur_onto, df_dur_stat)
    df_fine_final = compare_fines(df_fine_onto, df_fine_stat)
    df_counts_ca, df_counts_infra = compare_counts(df_counts, df_author_gb)

    # ── Plots ────────────────────────────────────────────────────────────
    generate_plots(
        df_age_final,
        df_dur_final,
        df_fine_final,
        df_counts_ca,
        df_counts_infra,
        plots_dir,
    )

    click.echo(f"Done. Plots saved to {plots_dir}")


if __name__ == "__main__":
    merge_csv()
