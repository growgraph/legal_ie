import logging
import sys

import pandas as pd
import pathlib
import click
import rdflib
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from legal_ie.util import extract_struct, render_response
import json

logger = logging.getLogger(__name__)


@click.command()
@click.option("--input-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--onto-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--env-path", type=click.Path(path_type=pathlib.Path), required=True)
def main(input_path, onto_path, env_path):
    input_path = input_path.as_posix()

    _ = load_dotenv(dotenv_path=env_path.expanduser())

    llm = ChatOpenAI(model="gpt-4o-mini")

    g_onto = rdflib.Graph()
    g_onto.parse(onto_path, format="turtle")

    onto_str = g_onto.serialize(format="turtle")

    with open(input_path) as f:
        query_json = json.load(f)

    query_text = query_json["text"]

    r = render_response(onto_str, text=query_text, llm=llm)
    extracted_text = extract_struct(r, "ttl")

    print(extracted_text)

    ttl_tent_fname = ".".join(input_path.split(".")[:-1] + ["tentative", "ttl"])

    with open(ttl_tent_fname, "w") as f:
        f.write(extracted_text)

    g = rdflib.Graph()
    _ = g.parse(data=extracted_text, format="turtle")

    current_ns = rdflib.Namespace("https://growgraph.dev/current#")
    specific_ns = rdflib.Namespace("http://example.org/criminal/case/0#")
    g.namespace_manager.bind(current_ns, specific_ns, replace=True)
    g.serialize(destination=".".join(input_path.split(".")[:-1] + ["ttl"]))
    triples = [[g.namespace_manager.normalizeUri(t) for t in tri] for tri in g]
    df = pd.DataFrame(triples, columns=["subject", "predicate", "object"]).sort_values(
        ["object", "subject", "predicate"]
    )

    csv_fname = ".".join(input_path.split(".")[:-1] + ["csv"])
    df.to_csv(csv_fname, index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()
