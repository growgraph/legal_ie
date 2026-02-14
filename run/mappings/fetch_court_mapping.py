"""
derive tribunal -- cour d'appel correspondence


"""

import pandas as pd
import pathlib
import click

BASE_URL = "https://www.data.gouv.fr/fr/datasets/liste-des-juridictions-competentes-pour-les-communes-de-france"


def extract_department(commune_insee):
    commune_insee = str(commune_insee).zfill(5)  # pad with zeros if needed

    if commune_insee.startswith("97") or commune_insee.startswith("98"):
        return commune_insee[:3]  # DOM-TOM
    elif commune_insee[:2] in {"2A", "2B"}:
        return commune_insee[:2]  # Corsica
    elif commune_insee.startswith("20"):
        # Ambiguity: old format for Corsica before 1976
        # Some datasets use 2A/2B; some still use 20
        return "2A or 2B"
    else:
        return commune_insee[:2]


@click.command()
@click.option("--csv-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--data-path", type=click.Path(path_type=pathlib.Path), required=True)
def main(csv_path, data_path):
    df = pd.read_csv(csv_path, sep=";", encoding="ISO-8859-1")
    df["Dept"] = df["Commune"].map(extract_department)
    df2 = df.drop_duplicates(
        ["Dept", "Cour d'Appel compétente", "Tribunal judiciaire compétent"]
    )
    df2[
        ["Dept", "Cour d'Appel compétente", "Tribunal judiciaire compétent"]
    ].sort_values("Dept").to_csv(data_path / "cour_tj_correspondence.csv", index=False)


if __name__ == "__main__":
    main()
