"""CLI for comparing ontology-extracted statistics with official crime statistics."""

import click
import numpy as np
from pathlib import Path

from legal_ie.plotting import generate_plots
from legal_ie.stat import (
    add_ministry_classification,
    compare_counts,
    compare_metric,
    derive_official_age,
    derive_ontology_counts,
    explode_and_group,
    iso_to_months,
    load_all_data,
    prepare_crime_type_filters,
    prepare_merged_data,
    prepare_official_stats,
    weighted_mean_by_group,
)


@click.command()
@click.option(
    "--data-dir",
    default="~/data/legal/working",
    help="Directory containing the CSV files",
    type=click.Path(),
)
@click.option(
    "--plots-dir",
    default="~/data/legal/plots",
    help="Directory to save comparison plots",
    type=click.Path(),
)
def run(data_dir, plots_dir):
    """Compare ontology-extracted statistics with official crime statistics."""
    data_dir = Path(data_dir).expanduser()
    plots_dir = Path(plots_dir).expanduser()

    # ── Load ─────────────────────────────────────────────────────────────
    data = load_all_data(data_dir)

    # ── Prepare merged ontology data ─────────────────────────────────────
    dft = prepare_merged_data(data["data"], data["mentions"], data["ca_tj"])

    # ── Ontology-side stats ──────────────────────────────────────────────
    # Per ca_canonical only (preserves data points without InfractionClass)
    df_fine_cc = explode_and_group(dft, "FineAmount")
    df_age_cc = explode_and_group(dft, "Age")
    df_dur_cc = explode_and_group(dft, "PunishmentDuration")

    # Per ca_canonical + OffenseType (InfractionClass-level, may drop rows)
    df_fine_cc_offense = explode_and_group(
        dft, "FineAmount", multi_groups=("OffenseType",)
    )
    df_age_cc_offense = explode_and_group(dft, "Age", multi_groups=("OffenseType",))
    df_dur_cc_offense = explode_and_group(
        dft, "PunishmentDuration", multi_groups=("OffenseType",)
    )

    df_counts = derive_ontology_counts(dft)

    # ── Official-side stats ──────────────────────────────────────────────
    df_crime_onto, df_gb_stat = prepare_crime_type_filters(data["crime_types_mapping"])
    df_author, df_conda, columns_nb = prepare_official_stats(
        data["stat_author"], data["stat_conda"], data["ca_tj"], df_crime_onto
    )

    df_author_mc, df_conda_mc, _ = prepare_official_stats(
        data["stat_author"], data["stat_conda"], data["ca_tj"], df_gb_stat
    )

    def dur_mean_fn(x):
        return np.mean([iso_to_months(d) for d in x]) / 12

    # -- InfractionClass level --
    df_dur_stat = weighted_mean_by_group(
        df_conda,
        "QMF_EMP_F",
        "NB_CONDA",
        ["ca_canonical", "InfractionClassOntologyClassification"],
    )
    df_fine_stat = weighted_mean_by_group(
        df_conda,
        "MMF_AME_F",
        "NB_CONDA",
        ["ca_canonical", "InfractionClassOntologyClassification"],
    )
    df_age_stat, df_author_gb = derive_official_age(df_author, columns_nb)

    # -- ca_canonical level only --
    df_dur_stat_ca = weighted_mean_by_group(
        df_conda, "QMF_EMP_F", "NB_CONDA", ["ca_canonical"]
    )
    df_fine_stat_ca = weighted_mean_by_group(
        df_conda, "MMF_AME_F", "NB_CONDA", ["ca_canonical"]
    )
    df_age_stat_ca, _ = derive_official_age(
        df_author, columns_nb, group_cols=("ca_canonical",)
    )

    # ── Ministry Classification perspective ──────────────────────────────
    _mc_col = "InfractionClassMinistryClassification"

    dft = add_ministry_classification(dft, df_gb_stat)

    df_fine_cc_mc = explode_and_group(
        dft, "FineAmount", multi_groups=("MinistryClassType",)
    )
    df_age_cc_mc = explode_and_group(dft, "Age", multi_groups=("MinistryClassType",))
    df_dur_cc_mc = explode_and_group(
        dft, "PunishmentDuration", multi_groups=("MinistryClassType",)
    )
    df_counts_mc = derive_ontology_counts(dft, offense_col="MinistryClassType")

    df_dur_stat_mc = weighted_mean_by_group(
        df_conda_mc, "QMF_EMP_F", "NB_CONDA", ["ca_canonical", _mc_col]
    )
    df_fine_stat_mc = weighted_mean_by_group(
        df_conda_mc, "MMF_AME_F", "NB_CONDA", ["ca_canonical", _mc_col]
    )
    df_age_stat_mc, df_author_mc_gb = derive_official_age(
        df_author_mc, columns_nb, group_cols=("ca_canonical", _mc_col)
    )

    # ── Comparisons ──────────────────────────────────────────────────────
    # ca_canonical level (all data points, no InfractionClass filter)
    df_age_by_ca = compare_metric(
        df_age_cc,
        df_age_stat_ca,
        "Age",
        np.mean,
        onto_mean_name="mean_age",
    )
    df_dur_by_ca = compare_metric(
        df_dur_cc,
        df_dur_stat_ca,
        "PunishmentDuration",
        dur_mean_fn,
        onto_mean_name="mean_dur_years",
        stat_name="dur_stat",
    )
    df_fine_by_ca = compare_metric(
        df_fine_cc,
        df_fine_stat_ca,
        "FineAmount",
        np.median,
        onto_mean_name="median_fine",
        stat_name="fine_stat",
    )

    # InfractionClass level (per OffenseType, matched with official categories)
    df_age_by_offense = compare_metric(
        df_age_cc_offense,
        df_age_stat,
        "Age",
        np.mean,
        onto_mean_name="mean_age",
    )
    df_dur_by_offense = compare_metric(
        df_dur_cc_offense,
        df_dur_stat,
        "PunishmentDuration",
        dur_mean_fn,
        onto_mean_name="mean_dur_years",
        stat_name="dur_stat",
    )
    df_fine_by_offense = compare_metric(
        df_fine_cc_offense,
        df_fine_stat,
        "FineAmount",
        np.mean,
        onto_mean_name="mean_fine",
        stat_name="fine_stat",
    )

    df_counts_ca, df_counts_infra = compare_counts(df_counts, df_author_gb)

    # Aggregate offense-level comparisons by InfractionClassOntologyClassification
    df_age_by_infra = df_age_by_offense.groupby(
        level="InfractionClassOntologyClassification"
    ).mean()
    df_dur_by_infra = df_dur_by_offense.groupby(
        level="InfractionClassOntologyClassification"
    ).mean()
    df_fine_by_infra = df_fine_by_offense.groupby(
        level="InfractionClassOntologyClassification"
    ).mean()

    # ── Comparisons (Ministry Classification) ────────────────────────────
    df_age_by_mc_offense = compare_metric(
        df_age_cc_mc,
        df_age_stat_mc,
        "Age",
        np.mean,
        onto_mean_name="mean_age",
        offense_type_col="MinistryClassType",
        infraction_col=_mc_col,
    )
    df_dur_by_mc_offense = compare_metric(
        df_dur_cc_mc,
        df_dur_stat_mc,
        "PunishmentDuration",
        dur_mean_fn,
        onto_mean_name="mean_dur_years",
        stat_name="dur_stat",
        offense_type_col="MinistryClassType",
        infraction_col=_mc_col,
    )
    df_fine_by_mc_offense = compare_metric(
        df_fine_cc_mc,
        df_fine_stat_mc,
        "FineAmount",
        np.mean,
        onto_mean_name="mean_fine",
        stat_name="fine_stat",
        offense_type_col="MinistryClassType",
        infraction_col=_mc_col,
    )
    _, df_counts_mc_infra = compare_counts(df_counts_mc, df_author_mc_gb)

    df_age_by_mc = df_age_by_mc_offense.groupby(level=_mc_col).mean()
    df_dur_by_mc = df_dur_by_mc_offense.groupby(level=_mc_col).mean()
    df_fine_by_mc = df_fine_by_mc_offense.groupby(level=_mc_col).mean()

    # ── Plots ─────────────────────────────────────────────────────────────
    generate_plots(
        df_age_by_ca,
        df_dur_by_ca,
        df_fine_by_ca,
        df_counts_ca,
        df_counts_infra,
        plots_dir,
        df_age_by_infra=df_age_by_infra,
        df_dur_by_infra=df_dur_by_infra,
        df_fine_by_infra=df_fine_by_infra,
        df_age_by_mc=df_age_by_mc,
        df_dur_by_mc=df_dur_by_mc,
        df_fine_by_mc=df_fine_by_mc,
        df_counts_mc_infra=df_counts_mc_infra,
    )

    click.echo(f"Done. Plots saved to {plots_dir}")


if __name__ == "__main__":
    run()
