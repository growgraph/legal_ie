import click
import pandas as pd
import numpy as np
from pathlib import Path
import re


def iso_to_months(duration_str):
    if not isinstance(duration_str, str):
        return 0

    total_months = 0
    # Find Years (e.g., P2Y -> 2 * 12)
    years = re.search(r"(\d+)Y", duration_str)
    if years:
        total_months += int(years.group(1)) * 12

    # Find Months (e.g., P6M or PT18M -> 6 or 18)
    # Note: In your data, PT18M is treated as 18 months
    months = re.search(r"(\d+)M", duration_str)
    if months:
        total_months += int(months.group(1))

    return total_months


def aggregate_to_set(df, key_col, value_cols):
    """
    Groups by `key_col` and aggregates all `value_cols` as sets.
    """
    agg_dict = {col: lambda x: tuple(sorted(set(x.dropna()))) for col in value_cols}
    return df.groupby(key_col, as_index=False).agg(agg_dict)


def dump_infra_mapping(df_crime_types_mapping):
    def decode_notation(row):
        n, left, right = row[["Notation", "Ontology_Class", "Ministry_Class"]].values
        if n == r"$\supset$":
            return f"({right}) is part of ({left})"
        elif n == r"$\subset$":
            return f"({left}) is part of ({right})"
        elif n == r"$\cap$":
            return f"({left}) and ({right}) have non-empty intersection"
        elif n == r"$\approx$":
            return f"({left}) and ({right}) are the same"
        return "error"

    df_tmp = (
        df_crime_types_mapping.loc[
            df_crime_types_mapping.Ministry_Class.notnull(),
            ["Ontology_Class", "Ministry_Class", "Notation"],
        ]
        .sort_values(["Ontology_Class", "Notation", "Ministry_Class"])
        .reset_index(drop=True)
    )

    df_tmp["Explanation"] = df_tmp.apply(decode_notation, axis=1)

    return df_tmp


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
def merge_csv(data_dir, output):
    """CLI to perform an outer join of case-age, case-fine-duration, and case-offense_type CSVs."""

    data_dir = Path(data_dir).expanduser()

    # Load CSVs
    data_file = data_dir / "case-age-fine-duration-offence-decision.csv"
    court_mentions_file = data_dir / "mentions.csv"
    ca_tj_file = data_dir / "bridge.csv"
    stat_author_file = data_dir / "AUTEUR_s.csv"
    stat_conda_file = data_dir / "CONDA_s.csv"
    crime_types_mapping_file = data_dir / "crime_types_cc_stat.csv"

    df_data = pd.read_csv(data_file)
    df_crime_types_mapping = pd.read_csv(crime_types_mapping_file, sep=";")

    df_data["AppealId"] = df_data["g"].apply(lambda x: x.split("/")[-1])
    df_data_set = aggregate_to_set(
        df_data, "AppealId", ["Age", "PunishmentDuration", "FineAmount", "OffenseType"]
    )

    df_mentions = pd.read_csv(court_mentions_file)
    df_mentions["AppealId"] = df_mentions["filename"].apply(
        lambda x: x.split("_")[1].split("°")[1]
    )
    df_mentions = (
        df_mentions.sort_values(["AppealId", "mention"])
        .groupby("AppealId", as_index=False)
        .first()
    )
    df_mentions["mention"] = df_mentions["mention"].str.lower()

    df_ca_tj = pd.read_csv(ca_tj_file)

    dft = pd.merge(df_data_set, df_mentions, on="AppealId")
    dft = pd.merge(
        dft,
        df_ca_tj[["ca_mention", "ca_canonical"]].drop_duplicates(
            ["ca_mention", "ca_canonical"]
        ),
        left_on="mention",
        right_on="ca_mention",
    )

    # age stats

    df_refined_age = dft.explode("OffenseType").explode("Age")
    df_refined_age = df_refined_age[
        df_refined_age["Age"].notnull() & df_refined_age["OffenseType"].notnull()
    ].copy()

    df_age_stat_non_empty = (
        df_refined_age.groupby(["ca_canonical", "OffenseType"])["Age"]
        .apply(np.array)
        .reset_index()
    )

    # counts
    df_counts = dft.explode("OffenseType")
    df_counts = df_counts.loc[df_counts["OffenseType"].notnull()].copy()
    df_counts = df_counts.groupby(["ca_canonical", "OffenseType"])["Age"].apply(
        lambda x: sum(len(xx) if xx else 1 for xx in x)
    )

    # fine stats
    df_refined_fine = dft.explode("OffenseType").explode("FineAmount")
    df_refined_fine = df_refined_fine[
        df_refined_fine["FineAmount"].notnull()
        & df_refined_fine["OffenseType"].notnull()
    ].copy()

    df_refined_fine = (
        df_refined_fine.groupby(["ca_canonical", "OffenseType"])["FineAmount"]
        .apply(np.array)
        .reset_index()
    )

    # duration stats
    df_refined_dur = dft.explode("OffenseType").explode("PunishmentDuration")
    df_refined_dur = df_refined_dur[
        df_refined_dur["PunishmentDuration"].notnull()
        & df_refined_dur["OffenseType"].notnull()
    ].copy()

    df_duration_non_empty = (
        df_refined_dur.groupby(["ca_canonical", "OffenseType"])["PunishmentDuration"]
        .apply(np.array)
        .reset_index()
    )

    df_stat_author = pd.read_csv(stat_author_file, sep=";")
    df_stat_conda = pd.read_csv(stat_conda_file, sep=";")
    df_stat_author = df_stat_author.drop(df_stat_author.columns[0], axis=1)
    df_stat_conda = df_stat_conda.drop(df_stat_conda.columns[0], axis=1)

    df_stat_author_canon = pd.merge(
        df_stat_author,
        df_ca_tj[["tj_canonical", "tj_stat", "ca_canonical"]].drop_duplicates(
            ["tj_canonical", "tj_stat"]
        ),
        left_on="LIB_JUR",
        right_on="tj_stat",
        how="left",
    )

    df_stat_conda_canon = pd.merge(
        df_stat_conda,
        df_ca_tj[["tj_canonical", "tj_stat", "ca_canonical"]].drop_duplicates(
            ["tj_canonical", "tj_stat"]
        ),
        left_on="LIB_JUR",
        right_on="tj_stat",
        how="left",
    )

    df_crime_types_mapping_notnull = df_crime_types_mapping.loc[
        df_crime_types_mapping.Ministry_Class.notnull(),
        df_crime_types_mapping.columns[:3],
    ].copy()
    df_crime_types_mapping_notnull_gb_onto = df_crime_types_mapping_notnull[
        df_crime_types_mapping_notnull["Notation"] != r"$\subset$"
    ].copy()
    df_crime_types_mapping_notnull_gb_stat = df_crime_types_mapping_notnull[
        df_crime_types_mapping_notnull["Notation"] != r"$\supset$"
    ].copy()

    df_crime_types_mapping_notnull_gb_stat.head()

    df_stat_author_canon_prep = pd.merge(
        df_stat_author_canon,
        df_crime_types_mapping_notnull_gb_onto.drop("Notation", axis=1),
        left_on="GR_INF",
        right_on="Ministry_Class",
        how="inner",
    )

    df_stat_conda_canon_prep = pd.merge(
        df_stat_conda_canon,
        df_crime_types_mapping_notnull_gb_onto.drop("Notation", axis=1),
        left_on="GR_INF",
        right_on="Ministry_Class",
        how="inner",
    )

    # df_stat_author_canon_prep_gb_ca = df_stat_conda_canon_prep.groupby(
    #     ["ca_canonical", "Ontology_Class"]
    # )[columns_nb].sum()
    #
    # df_stat_conda_canon_prep["QMF_EMP_F"]

    columns_nb = [c for c in df_stat_author.columns if c.startswith("NB")]
    for c in columns_nb:
        df_stat_author_canon_prep[c] = (
            df_stat_author_canon_prep[c].replace("<5", 2).replace("nc", 5).astype(int)
        )

    df_stat_conda_canon_prep["NB_CONDA"] = (
        df_stat_conda_canon_prep["NB_CONDA"]
        .replace("<5", 2)
        .replace("nc", 5)
        .astype(int)
    )
    df_emp_dur_stat = df_stat_conda_canon_prep.groupby(
        ["ca_canonical", "Ontology_Class"]
    ).apply(lambda x: np.dot(x["QMF_EMP_F"], x["NB_CONDA"]) / x["NB_CONDA"].sum())

    df_fine_stat = df_stat_conda_canon_prep.groupby(
        ["ca_canonical", "Ontology_Class"]
    ).apply(lambda x: np.dot(x["MMF_AME_F"], x["NB_CONDA"]) / x["NB_CONDA"].sum())

    df_fine_stat, df_emp_dur_stat
    df_stat_author_canon_prep_gb_ca = df_stat_author_canon_prep.groupby(
        ["ca_canonical", "Ontology_Class"]
    )[columns_nb].sum()

    columns_ages = [c for c in columns_nb if c.split("_")[1][0].isnumeric()]

    # no need since we are working only with means
    # we subtract 0.05 so that e.g. age 26 would always be found in second bin (index 1), if the ages are integers
    # in other words not to worry about strict/non-strict inequalities
    bins = np.array([float(c.split("_")[1][:2]) for c in columns_ages] + [120.0])

    age_group_centers = [np.mean(x) for x in zip(bins, bins[1:])]

    df_age_stat = (
        df_stat_author_canon_prep_gb_ca.apply(
            lambda x: np.dot(x[columns_ages], age_group_centers)
            / x[columns_ages].sum(),
            axis=1,
        )
        .rename("mean_age_stat")
        .to_frame()
    )
    df_age = df_age_stat_non_empty.rename(columns={"OffenseType": "Ontology_Class"})
    df_age["mean_age"] = df_age["Age"].apply(np.mean)

    df_age_final = pd.merge(
        df_age[["ca_canonical", "Ontology_Class", "mean_age"]],
        df_age_stat,
        left_on=["ca_canonical", "Ontology_Class"],
        right_index=True,
    )

    df_age_final.head()
    ## counts

    df_counts_ca = pd.merge(
        df_stat_author_canon_prep_gb_ca["NB_AUT"]
        .groupby(level=0)
        .sum()
        .rename("cnt_ca_stat"),
        df_counts.groupby(level=0).sum().rename("cnt_ca"),
        left_index=True,
        right_index=True,
    )
    df_counts_ca["dist_ca_stat"] = (
        df_counts_ca["cnt_ca_stat"] / df_counts_ca["cnt_ca_stat"].sum()
    )
    df_counts_ca["dist_ca"] = df_counts_ca["cnt_ca"] / df_counts_ca["cnt_ca"].sum()

    df_counts_infra = pd.merge(
        df_stat_author_canon_prep_gb_ca["NB_AUT"]
        .groupby(level=1)
        .sum()
        .rename("cnt_infra_stat"),
        df_counts.groupby(level=1).sum().rename("cnt_infra"),
        left_index=True,
        right_index=True,
    )
    df_counts_infra["dist_infra_stat"] = (
        df_counts_infra["cnt_infra_stat"] / df_counts_infra["cnt_infra_stat"].sum()
    )
    df_counts_infra["dist_infra"] = (
        df_counts_infra["cnt_infra"] / df_counts_infra["cnt_infra"].sum()
    )

    # imprisonment duration

    df_duration_non_empty["mean_dur_months"] = df_duration_non_empty[
        "PunishmentDuration"
    ].apply(lambda x: np.mean([iso_to_months(d) for d in x]) / 12)

    df_dur_final = pd.merge(
        df_emp_dur_stat.rename("dur_stat").to_frame(),
        df_duration_non_empty.rename(columns={"OffenseType": "Ontology_Class"})
        .set_index(["ca_canonical", "Ontology_Class"])
        .mean_dur_months.to_frame(),
        left_index=True,
        right_index=True,
    )

    df_dur_final.head()
    # dfexp = dump_infra_mapping(df_crime_types_mapping)
    # dfexp.to_csv("~/data/legal/infra_types.csv", index=False)

    click.echo("Merged")


if __name__ == "__main__":
    merge_csv()
