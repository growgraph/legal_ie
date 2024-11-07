import logging
import sys

import click
import pandas as pd
import rdflib


from legal_ie.util import visualize, visualize_nx

logger = logging.getLogger(__name__)


@click.command()
@click.option("--ttl-input-path", type=click.Path(), required=True)
def main(ttl_input_path):
    g = rdflib.Graph()
    g.parse(ttl_input_path, format="turtle")

    triples_onto = [[g.namespace_manager.normalizeUri(t) for t in tri] for tri in g]

    df = pd.DataFrame(
        triples_onto, columns=["subject", "predicate", "object"]
    ).sort_values(["object", "subject", "predicate"])

    csv_fname = ".".join(ttl_input_path.split(".")[:-1] + ["csv"])
    df.to_csv(csv_fname, index=None)

    visualize(g, ".".join(ttl_input_path.split(".")[:-1] + ["png"]))

    mask_trivial = df.predicate.isin(["rdfs:label", "rdfs:comment"]) | df.object.isin(
        ["rdf:Property", "rdfs:Class"]
    )
    df_reduced = df.loc[~mask_trivial]

    visualize_nx(
        df_reduced,
        ".".join(ttl_input_path.split(".")[:-1] + ["nx.png"]),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()
