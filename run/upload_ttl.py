import re
import requests
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import os
import pathlib
import click
from dotenv import load_dotenv
from tqdm import tqdm

import logging
from legal_ie.lake import crawl_directories

logger = logging.getLogger(__name__)

GRAPH_BASE = "http://growgraph.dev/appeal"


def extract_case_id(filename: str) -> str | None:
    """Extract case ID from filename.

    The case ID sits between '°' and the next '_'.
    E.g. 'pourvoi_n°22-84.524_08_02_2023.ttl' -> '22-84.524'
    """
    match = re.search(r"°(.+?)_", filename)
    if match:
        return match.group(1)
    return None


def case_id_to_graph_uri(case_id: str) -> str:
    """Turn a case ID into a named-graph URI."""
    return f"{GRAPH_BASE}/{case_id}"


def init_dataset(dataset_name: str):
    username = os.environ["TS_USERNAME"]
    password = os.environ["TS_PASSWORD"]
    port = os.environ["TS_PORT"]
    fuseki_admin_url = f"http://localhost:{port}/$/datasets"

    payload = {"dbName": dataset_name, "dbType": "tdb2"}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = requests.post(
        fuseki_admin_url, data=payload, headers=headers, auth=(username, password)
    )

    if response.status_code in (200, 201):
        logger.info(f"Dataset '{dataset_name}' created successfully.")
    else:
        logger.error(f"Failed to create dataset. Status code: {response.status_code}")
        logger.error(f"Response: {response.text}")


def upload_ttl_to_fuseki(
    file_path: pathlib.Path, dataset: str, graph_uri: str | None = None
) -> bool:
    """Upload a TTL file to Fuseki via the Graph Store Protocol.

    If *graph_uri* is given the data is PUT into that named graph
    (replacing any previous content for the graph).  Otherwise the
    data goes into the default graph.
    """
    username = os.environ["TS_USERNAME"]
    password = os.environ["TS_PASSWORD"]
    port = os.environ["TS_PORT"]
    fuseki_url = f"http://localhost:{port}/{dataset}/data"

    params = {"graph": graph_uri} if graph_uri is not None else {"default": ""}
    headers = {"Content-Type": "text/turtle;charset=utf-8"}

    with open(file_path, "r") as fh:
        data = fh.read()

    response = requests.put(
        fuseki_url,
        headers=headers,
        params=params,
        data=data,
        auth=(username, password),
    )

    if response.status_code in (200, 201, 204):
        return True

    logger.error(
        f"Failed to upload {file_path.name}. Status code: {response.status_code}"
    )
    logger.error(f"Response: {response.text}")
    return False


def upload_parallel(
    pairs: list[tuple[pathlib.Path, str | None]],
    dataset: str,
    workers: int = 4,
) -> int:
    """Upload *pairs* of (file, graph_uri) in parallel with a progress bar.

    Returns the number of successful uploads.
    """
    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(upload_ttl_to_fuseki, fp, dataset, graph_uri): fp
            for fp, graph_uri in pairs
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Uploading",
            unit="file",
        ):
            fp = futures[future]
            try:
                if future.result():
                    ok += 1
            except Exception:
                logger.exception(f"Unexpected error uploading {fp.name}")
    return ok


@click.command()
@click.option("--input-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--onto-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--env-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--dataset", type=click.STRING, default="test")
@click.option("--head", type=click.INT, default=None)
@click.option("--workers", type=click.INT, default=4, help="Upload threads.")
def main(input_path, onto_path, env_path, dataset, head, workers):
    input_path = input_path.expanduser()
    onto_path = onto_path.expanduser()
    _ = load_dotenv(dotenv_path=env_path.expanduser())

    init_dataset(dataset)

    # -- ontology goes into the default graph --------------------------------
    logger.info(f"Uploading ontology {onto_path.name} into default graph")
    upload_ttl_to_fuseki(onto_path, dataset, graph_uri=None)

    # -- case files go into per-case named graphs ----------------------------
    files = sorted(crawl_directories(input_path.expanduser(), suffixes=(".ttl",)))
    if head is not None:
        files = files[:head]

    pairs: list[tuple[pathlib.Path, str | None]] = []
    for fp in files:
        case_id = extract_case_id(fp.name)
        if case_id is None:
            logger.warning(f"Could not extract case ID from {fp.name}, skipping")
            continue
        pairs.append((fp, case_id_to_graph_uri(case_id)))

    logger.info(f"Uploading {len(pairs)} case files as named graphs")
    ok = upload_parallel(pairs, dataset, workers=workers)
    logger.info(f"Done: {ok}/{len(pairs)} uploaded successfully")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()
