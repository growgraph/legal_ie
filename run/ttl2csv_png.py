import logging
import sys

import click
import pandas as pd
import rdflib

import pathlib
from legal_ie.util import visualize, visualize_nx

logger = logging.getLogger(__name__)


def proc(ttl_fname, plot):
    g = rdflib.Graph()
    g.parse(ttl_fname, format="turtle")

    triples_onto = [[g.namespace_manager.normalizeUri(t) for t in tri] for tri in g]

    df = pd.DataFrame(
        triples_onto, columns=["subject", "predicate", "object"]
    ).sort_values(["object", "subject", "predicate"])

    csv_fname = ".".join(ttl_fname.split(".")[:-1] + ["csv"])
    df.sort_values(["object", "subject", "predicate"]).to_csv(
        csv_fname, index=None, encoding="utf-8"
    )

    if plot:
        visualize(g, ".".join(ttl_fname.split(".")[:-1] + ["png"]))

        mask_trivial = df.predicate.isin(
            ["rdfs:label", "rdfs:comment"]
        ) | df.object.isin(["rdf:Property", "rdfs:Class"])
        df_reduced = df.loc[~mask_trivial]

        visualize_nx(
            df_reduced, ".".join(ttl_fname.split(".")[:-1] + ["nx.png"]), scale=1.1
        )


@click.command()
@click.option(
    "--ttl-input-path", type=click.Path(path_type=pathlib.Path), required=True
)
@click.option(
    "--plot",
    type=click.BOOL,
    is_flag=True,
    default=False,
    help="plot ttls",
)
def main(ttl_input_path, plot):
    if ttl_input_path.expanduser().is_dir():
        ttls = [
            f
            for f in ttl_input_path.expanduser().iterdir()
            if f.suffix == ".ttl" and f.is_file() and "tentative" not in f.name
        ]
    else:
        ttls = [ttl_input_path] if ttl_input_path.suffix == ".ttl" else []

    for ttl in ttls:
        proc(ttl.as_posix(), plot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()
