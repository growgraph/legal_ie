"""
Build the bridge mapping:

    CA(mentions) <-> CA(correspondence) <-> TJ(correspondence) <-> TJ(stat_s)

Inputs
------
- cour_tj_correspondence.csv   — canonical CA <-> TJ pairs (comma-separated)
- mentions.csv                 — surface forms of Cour d'Appel from case PDFs
- <stat>_s.csv                 — surface forms of Tribunal Judiciaire from stats
                                 (semicolon-separated), e.g. AUTEUR.csv

Outputs
-------
- ca_map.csv      — fuzzy match:  CA(mentions) -> CA(correspondence)
- tj_map.csv      — fuzzy match:  TJ(stat_s) -> TJ(correspondence)
- bridge.csv      — final bridge: CA(mentions) <-> TJ(stat_s)
                     via the correspondence table
"""

import pathlib

import click
import pandas as pd

from legal_ie.mapping import create_location_mapping


COL_CA = "Cour d'Appel compétente"
COL_TJ = "Tribunal judiciaire compétent"


def _load_correspondence(path: pathlib.Path) -> pd.DataFrame:
    """Load the canonical CA <-> TJ correspondence table."""
    df = pd.read_csv(path)
    assert COL_CA in df.columns and COL_TJ in df.columns, (
        f"Expected columns '{COL_CA}' and '{COL_TJ}' in {path}"
    )
    return df


def _load_mentions(path: pathlib.Path) -> list[str]:
    """Return sorted unique lower-cased CA mentions."""
    dfm = pd.read_csv(path)
    dfm["mention"] = dfm["mention"].str.lower().str.strip()
    return sorted(dfm["mention"].unique())


def _load_stat(path: pathlib.Path) -> list[str]:
    """Return sorted unique TJ labels from stat_s.csv (semicolon-separated)."""
    dfa = pd.read_csv(path, sep=";")
    return sorted(dfa["LIB_JUR"].dropna().unique())


@click.command()
@click.option(
    "--corr-path",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    help="Path to cour_tj_correspondence.csv",
)
@click.option(
    "--mentions-path",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    help="Path to mentions.csv (CA surface forms)",
)
@click.option(
    "--stat-path",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    help="Path to stat_s.csv, e.g. AUTEUR or CONDA csv (TJ surface forms, semicolon-separated)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    help="Directory to write ca_map.csv, tj_map.csv, and bridge.csv",
)
@click.option(
    "--threshold", default=60, show_default=True, help="Fuzzy match threshold (0-100)"
)
def main(
    corr_path: pathlib.Path,
    mentions_path: pathlib.Path,
    stat_path: pathlib.Path,
    output_dir: pathlib.Path,
    threshold: int,
):
    """
    Build the bridge mapping CA(mentions) <-> TJ(stat_s)
    via the cour_tj_correspondence table.
    """
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load data -----------------------------------------------------------
    corr = _load_correspondence(corr_path.expanduser())
    ca_mentions = _load_mentions(mentions_path.expanduser())
    tj_stat = _load_stat(stat_path.expanduser())

    ca_canonical = sorted(corr[COL_CA].unique())
    tj_canonical = sorted(corr.loc[corr[COL_TJ].notnull(), COL_TJ].unique())

    click.echo(
        f"Loaded {len(ca_mentions)} CA mentions, {len(tj_stat)} TJ labels, "
        f"{len(ca_canonical)} canonical CAs, {len(tj_canonical)} canonical TJs"
    )

    # --- 2. Fuzzy-match CA(mentions) -> CA(correspondence) ----------------------
    click.echo("\n=== Matching CA mentions -> CA canonical ===")
    ca_map = create_location_mapping(
        ca_mentions, ca_canonical, threshold=threshold, auto_detect_prefixes=True
    )
    ca_map_path = output_dir / "ca_map.csv"
    ca_map.sort_values("similarity_score", ascending=True).to_csv(
        ca_map_path, index=False
    )
    click.echo(
        f"  Matched {len(ca_map)}/{len(ca_mentions)} CA mentions -> {ca_map_path}"
    )

    unmatched_ca = set(ca_mentions) - set(ca_map["left_o"])
    if unmatched_ca:
        click.echo(f"  Unmatched CA mentions ({len(unmatched_ca)}): {unmatched_ca}")

    # --- 3. Fuzzy-match TJ(stat_s) -> TJ(correspondence) ---------------------
    click.echo("\n=== Matching TJ stat labels -> TJ canonical ===")
    tj_map = create_location_mapping(
        tj_stat, tj_canonical, threshold=threshold, auto_detect_prefixes=True
    )
    tj_map_path = output_dir / "tj_map.csv"
    tj_map.sort_values("similarity_score", ascending=True).to_csv(
        tj_map_path, index=False
    )
    click.echo(f"  Matched {len(tj_map)}/{len(tj_stat)} TJ labels -> {tj_map_path}")

    unmatched_tj = set(tj_stat) - set(tj_map["left_o"])
    if unmatched_tj:
        click.echo(f"  Unmatched TJ labels ({len(unmatched_tj)}): {unmatched_tj}")

    # --- 4. Build bridge: CA(mentions) <-> TJ(stat_s) via correspondence ------
    click.echo("\n=== Building bridge mapping ===")

    # ca_map:  left_o = CA(mention),  right_o = CA(canonical)
    # tj_map:  left_o = TJ(stat),   right_o = TJ(canonical)
    # corr:    COL_CA = CA(canonical), COL_TJ = TJ(canonical)

    # Step A: join ca_map with correspondence on canonical CA
    bridge = pd.merge(
        ca_map[["left_o", "right_o", "similarity_score"]].rename(
            columns={
                "left_o": "ca_mention",
                "right_o": "ca_canonical",
                "similarity_score": "ca_score",
            }
        ),
        corr[[COL_CA, COL_TJ]].drop_duplicates(),
        left_on="ca_canonical",
        right_on=COL_CA,
        how="inner",
    )

    # Step B: join with tj_map on canonical TJ
    bridge = pd.merge(
        bridge,
        tj_map[["left_o", "right_o", "similarity_score"]].rename(
            columns={
                "left_o": "tj_stat",
                "right_o": "tj_canonical",
                "similarity_score": "tj_score",
            }
        ),
        left_on=COL_TJ,
        right_on="tj_canonical",
        how="inner",
    )

    # Keep the useful columns
    bridge = bridge[
        [
            "ca_mention",
            "ca_canonical",
            "ca_score",
            "tj_canonical",
            "tj_stat",
            "tj_score",
        ]
    ].sort_values(["ca_mention", "tj_stat"])

    bridge_path = output_dir / "bridge.csv"
    bridge.to_csv(bridge_path, index=False)
    click.echo(
        f"  Bridge mapping: {len(bridge)} rows "
        f"({bridge['ca_mention'].nunique()} CAs x {bridge['tj_stat'].nunique()} TJs) "
        f"-> {bridge_path}"
    )


if __name__ == "__main__":
    main()
