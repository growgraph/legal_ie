from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd


def generate_plots(
    df_age_final: pd.DataFrame,
    df_dur_final: pd.DataFrame,
    df_fine_final: pd.DataFrame,
    df_counts_ca: pd.DataFrame,
    df_counts_infra: pd.DataFrame,
    plots_dir: Path,
    *,
    df_age_by_infra: pd.DataFrame | None = None,
    df_dur_by_infra: pd.DataFrame | None = None,
    df_fine_by_infra: pd.DataFrame | None = None,
    df_age_by_mc: pd.DataFrame | None = None,
    df_dur_by_mc: pd.DataFrame | None = None,
    df_fine_by_mc: pd.DataFrame | None = None,
    df_counts_mc_infra: pd.DataFrame | None = None,
) -> None:
    """Save all comparison plots to *plots_dir*."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Age: parity plot
    plot_parity(
        df_age_final.reset_index(),
        "mean_age",
        "mean_age_stat",
        xlabel="Mean age (Cour de Cassation stats)",
        ylabel="Mean age (Overall stats)",
        title="Age: Cour de Cassation stats vs Global statistics",
        output_path=plots_dir / "age_comparison.png",
        annotation_col="ca_canonical",
    )

    # Duration: parity plot
    plot_parity(
        df_dur_final.reset_index(),
        "mean_dur_years",
        "dur_stat",
        xlabel="Mean duration (Cour de Cassation stats)",
        ylabel="Mean duration (Overall stats)",
        title="Imprisonment duration: Cour de Cassation stats vs Global statistics",
        output_path=plots_dir / "duration_comparison.png",
        annotation_col="ca_canonical",
    )

    # Fine: parity plot
    plot_parity(
        df_fine_final.reset_index(),
        "median_fine",
        "fine_stat",
        xlabel="Median fine (Cour de Cassation stats)",
        ylabel="Median fine (Overall stats)",
        title="Fine amount: Cour de Cassation stats vs Global statistics",
        output_path=plots_dir / "fine_comparison.png",
        logx=True,
        logy=True,
        annotation_col="ca_canonical",
    )

    # ── Parity plots by InfractionClass ───────────────────────────────
    _infra_col = "InfractionClassOntologyClassification"

    if df_age_by_infra is not None:
        plot_parity(
            df_age_by_infra.reset_index(),
            "mean_age",
            "mean_age_stat",
            xlabel="Mean age (CC stats)",
            ylabel="Mean age (Official stats)",
            title="Age: CC stats vs Official stats (by Infraction Class)",
            output_path=plots_dir / "age_by_infraction_class.png",
            annotation_col=_infra_col,
        )

    if df_dur_by_infra is not None:
        plot_parity(
            df_dur_by_infra.reset_index(),
            "mean_dur_years",
            "dur_stat",
            xlabel="Mean duration in years (CC stats)",
            ylabel="Mean duration in years (Official stats)",
            title="Duration: CC stats vs Official stats (by Infraction Class)",
            output_path=plots_dir / "duration_by_infraction_class.png",
            annotation_col=_infra_col,
        )

    if df_fine_by_infra is not None:
        plot_parity(
            df_fine_by_infra.reset_index(),
            "mean_fine",
            "fine_stat",
            xlabel="Mean fine (CC stats)",
            ylabel="Mean fine (Official stats)",
            title="Fine: CC stats vs Official stats (by Infraction Class)",
            output_path=plots_dir / "fine_by_infraction_class.png",
            logx=True,
            logy=True,
            annotation_col=_infra_col,
        )

    # Geographic distribution
    plot_distribution_comparison(
        labels=df_counts_ca.index.tolist(),
        values_a=df_counts_ca["dist_ca"].values,
        values_b=df_counts_ca["dist_ca_stat"].values,
        name_a="Cour de Cassation stats",
        name_b="Official Stats",
        title="Geographic distribution of cases",
        output_path=plots_dir / "geography_distribution.png",
    )

    # Crime-type distribution
    plot_distribution_comparison(
        labels=df_counts_infra.index.tolist(),
        values_a=df_counts_infra["dist_infra"].values,
        values_b=df_counts_infra["dist_infra_stat"].values,
        name_a="Cour de Cassation stats",
        name_b="Official Stats",
        title="Crime-type distribution",
        output_path=plots_dir / "crime_type_distribution.png",
    )

    # ── Ministry Classification perspective ──────────────────────────────
    _mc_col = "InfractionClassMinistryClassification"

    if df_age_by_mc is not None:
        plot_parity(
            df_age_by_mc.reset_index(),
            "mean_age",
            "mean_age_stat",
            xlabel="Mean age (CC stats)",
            ylabel="Mean age (Official stats)",
            title="Age: CC stats vs Official stats (by Ministry Classification)",
            output_path=plots_dir / "age_by_ministry_class.png",
            annotation_col=_mc_col,
        )

    if df_dur_by_mc is not None:
        plot_parity(
            df_dur_by_mc.reset_index(),
            "mean_dur_years",
            "dur_stat",
            xlabel="Mean duration in years (CC stats)",
            ylabel="Mean duration in years (Official stats)",
            title="Duration: CC stats vs Official stats (by Ministry Classification)",
            output_path=plots_dir / "duration_by_ministry_class.png",
            annotation_col=_mc_col,
        )

    if df_fine_by_mc is not None:
        plot_parity(
            df_fine_by_mc.reset_index(),
            "mean_fine",
            "fine_stat",
            xlabel="Mean fine (CC stats)",
            ylabel="Mean fine (Official stats)",
            title="Fine: CC stats vs Official stats (by Ministry Classification)",
            output_path=plots_dir / "fine_by_ministry_class.png",
            logx=True,
            logy=True,
            annotation_col=_mc_col,
        )

    if df_counts_mc_infra is not None:
        plot_distribution_comparison(
            labels=df_counts_mc_infra.index.tolist(),
            values_a=df_counts_mc_infra["dist_infra"].values,
            values_b=df_counts_mc_infra["dist_infra_stat"].values,
            name_a="Cour de Cassation stats",
            name_b="Official Stats",
            title="Crime-type distribution (Ministry Classification)",
            output_path=plots_dir / "crime_type_distribution_mc.png",
        )


def plot_parity(
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    output_path: Path | None = None,
    logx: bool = False,
    logy: bool = False,
    annotation_col: str | None = None,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure


    def process_label(x):
        if annotation_col == "ca_canonical":
            return x.split(" ")[-1]
        elif annotation_col == "InfractionClassMinistryClassification":
            return " ".join(x.split(" ")[1:])
        return x

    if annotation_col is not None:
        labels = df[annotation_col].unique()
        palette = sns.color_palette("husl", n_colors=len(labels))
        color_map = dict(zip(labels, palette))

        for label in labels:
            subset = df[df[annotation_col] == label]
            color = color_map[label]
            ax.scatter(
                subset[col_x],
                subset[col_y],
                color=color,
                label=process_label(label),
                alpha=0.6,
                edgecolors="k",
                linewidths=0.5,
            )
            for _, row in subset.iterrows():
                ax.annotate(
                    process_label(label),
                    (row[col_x], row[col_y]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=7,
                    color=color,
                    alpha=0.85,
                )
        ax.legend(
            fontsize=7,
            title=annotation_col,
            loc="best",
            framealpha=0.8,
        )
    else:
        ax.scatter(df[col_x], df[col_y], alpha=0.6, edgecolors="k", linewidths=0.5)

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    lo = min(df[col_x].min(), df[col_y].min())
    hi = max(df[col_x].max(), df[col_y].max())

    if logx and logy:
        lo_lim = 10.0 ** int(np.floor(np.log10(lo)))
        hi_lim = 10.0 ** int(np.ceil(np.log10(hi)))
    else:
        margin = (hi - lo) * 0.05 or 1.0
        lo_lim = lo - margin
        hi_lim = hi + margin

    ax.plot([lo_lim, hi_lim], [lo_lim, hi_lim], ls="--", color="grey", linewidth=1)
    ax.set_xlim(lo_lim, hi_lim)
    ax.set_ylim(lo_lim, hi_lim)
    ax.set_aspect("equal", adjustable="box")

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
    sort_by: str = "a",
):
    """Bar chart comparing two distributions side by side.

    Parameters
    ----------
    sort_by : str
        ``"a"`` (default) sorts descending by *values_a*,
        ``"b"`` sorts descending by *values_b*,
        ``"none"`` keeps the original order.
    """
    labels = np.asarray(labels)
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)

    if sort_by == "a":
        order = np.argsort(-values_a)
    elif sort_by == "b":
        order = np.argsort(-values_b)
    else:
        order = np.arange(len(labels))

    labels = labels[order]
    values_a = values_a[order]
    values_b = values_b[order]

    x = np.arange(len(labels))
    width = 0.35

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 6))
    else:
        fig = ax.figure

    color_a, color_b = sns.color_palette("Set2", 2)

    ax.bar(
        x - width / 2,
        values_a,
        width,
        label=name_a,
        color=color_a,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
    )
    ax.bar(
        x + width / 2,
        values_b,
        width,
        label=name_b,
        color=color_b,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate_labels, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    sns.despine(ax=ax, left=True)

    if output_path:
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    return ax
