import logging
import sys

import pathlib
import click
import rdflib
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from legal_ie.lake import crawl_directories, process_file
from legal_ie.util import render_response
from datetime import datetime
import json
from multiprocessing.pool import ThreadPool
from legal_ie.util import extract_struct, replace_namespaces

logger = logging.getLogger(__name__)


def process_unit(rel_fp, text, onto_str, output_path, fresh=False):
    llm = ChatOpenAI(model="gpt-4o-mini")

    output_path = output_path.expanduser()
    opath = output_path / pathlib.Path(*rel_fp.parts[:-1])
    if not opath.exists():
        opath.mkdir(parents=True, exist_ok=True)

    fn_json_raw = (output_path / rel_fp).with_suffix(".json")
    fn_ttl = (output_path / rel_fp).with_suffix(".ttl")
    fn_csv = (output_path / rel_fp).with_suffix(".csv")
    fn_json_comments = (output_path / rel_fp).with_suffix(".comments.json")

    # example "pourvoi_n°22-01.836_02_01_2023.pdf"
    tmp = rel_fp.stem.split("°")
    if len(tmp) == 1:
        tmp_str = tmp[0]
    else:
        tmp_str = tmp[1]

    tmp = tmp_str.split("_")
    doc_id = tmp[0].replace(".", "--")

    if fn_ttl.exists() and not fresh:
        return False
    else:
        response_raw = render_response(onto_str, text=text, llm=llm)

        extracted_ttl = extract_struct(response_raw, "ttl")
        extracted_comments = extract_struct(response_raw, "remarks")
        with open(fn_json_raw, "w") as outfile:
            json.dump(
                {"raw": response_raw, "date": datetime.now().isoformat()},
                outfile,
                indent=4,
            )

        with open(fn_json_comments, "w") as outfile:
            json.dump({"remarks": extracted_comments}, outfile, indent=4)

        g = rdflib.Graph()
        _ = g.parse(data=extracted_ttl, format="turtle")

        current_ns = rdflib.Namespace("https://growgraph.dev/current#")
        specific_ns = rdflib.Namespace(f"https://growgraph.dev/fca/appeal/{doc_id}#")

        g = replace_namespaces(g, current_ns, specific_ns)

        g.serialize(destination=fn_ttl)

        triples = [[g.namespace_manager.normalizeUri(t) for t in tri] for tri in g]
        df = pd.DataFrame(
            triples, columns=["subject", "predicate", "object"]
        ).sort_values(["object", "subject", "predicate"])

        df.to_csv(fn_csv, index=False)
    return True


def process_batch(batch, onto_str, output_path, fresh, to_kg=True):
    ts = [(rfp, *process_file(fp), onto_str, output_path, fresh) for fp, rfp in batch]
    if to_kg:
        if len(ts) > 1:
            with ThreadPool(processes=4) as pool:
                _ = pool.map(process_unit, ts)
        else:
            for item in ts:
                process_unit(*item)


@click.command()
@click.option("--input-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--output-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--onto-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--env-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--head", type=click.INT, default=None)
@click.option("--batch-size", type=click.INT, default=10)
@click.option("--fresh", type=click.BOOL, is_flag=True, default=False)
def main(input_path, output_path, onto_path, env_path, head, batch_size, fresh):
    input_path = input_path.expanduser()
    _ = load_dotenv(dotenv_path=env_path.expanduser())

    g_onto = rdflib.Graph()
    g_onto.parse(onto_path, format="turtle")

    onto_str = g_onto.serialize(format="turtle")

    nabs_parts = len(input_path.expanduser().parts)

    files = sorted(crawl_directories(input_path.expanduser()))
    files = [(fp, pathlib.Path(*fp.parts[nabs_parts:])) for fp in files]
    if head is not None:
        files = files[:head]

    batches = [
        files[i * batch_size : (i + 1) * batch_size]
        for i in range(int(len(files) / batch_size) + 1)
    ]

    for batch in batches:
        process_batch(batch, onto_str, output_path, fresh)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()
