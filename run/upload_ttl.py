import requests
import sys

import os
import pathlib
import click
from dotenv import load_dotenv

import logging
from legal_ie.lake import crawl_directories
from multiprocessing.pool import ThreadPool

logger = logging.getLogger(__name__)


def init_dataset(dataset_name):
    username = os.environ["TS_USERNAME"]
    password = os.environ["TS_PASSWORD"]
    port = os.environ["TS_PORT"]
    fuseki_admin_url = f"http://localhost:{port}/$/datasets"

    payload = {"dbName": dataset_name, "dbType": "tdb2"}

    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = requests.post(
        fuseki_admin_url, data=payload, headers=headers, auth=(username, password)
    )

    if response.status_code == 200 or response.status_code == 201:
        logger.info(f"Dataset '{dataset_name}' created successfully.")
    else:
        logger.error(f"Failed to upload data. Status code: {response.status_code}")
        logger.error(f"Response: {response.text}")


def upload_ttl_to_fuseki(file_path, dataset):
    username = os.environ["TS_USERNAME"]
    password = os.environ["TS_PASSWORD"]
    port = os.environ["TS_PORT"]
    fuseki_url = f"http://localhost:{port}/{dataset}"

    headers = {"Content-Type": "text/turtle;charset=utf-8"}

    with open(file_path, "r") as file:
        data = file.read()

    response = requests.post(
        f"{fuseki_url}/data", headers=headers, data=data, auth=(username, password)
    )

    if response.status_code == 200:
        return True
    else:
        logger.error(f"Failed to upload data. Status code: {response.status_code}")
        logger.error(f"Response: {response.text}")
        return False


def process_batch(batch, dataset):
    if len(batch) > 1:
        with ThreadPool(processes=4) as pool:
            flags = pool.starmap(
                upload_ttl_to_fuseki, zip(batch, [dataset] * len(batch))
            )
    else:
        flags = [upload_ttl_to_fuseki(item, dataset) for item in batch]
    logger.info(f" flags : {sum(flags)}/{len(flags)}")


@click.command()
@click.option("--input-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--onto-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--env-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--dataset", type=click.STRING, default="fca2")
@click.option("--head", type=click.INT, default=None)
@click.option("--batch-size", type=click.INT, default=10)
def main(input_path, onto_path, env_path, dataset, head, batch_size):
    input_path = input_path.expanduser()
    onto_path = onto_path.expanduser()
    _ = load_dotenv(dotenv_path=env_path.expanduser())

    init_dataset(dataset)

    files = sorted(crawl_directories(input_path.expanduser(), suffixes=(".ttl",)))
    if head is not None:
        files = files[:head]

    files = [onto_path] + files
    batches = [
        files[i * batch_size : (i + 1) * batch_size]
        for i in range(int(len(files) / batch_size) + 1)
    ]

    for batch in batches:
        process_batch(batch, dataset)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()
