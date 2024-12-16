import pandas as pd
import pathlib
import click
import logging.config

from legal_ie.scrape import process_date

logger = logging.getLogger(__name__)


@click.command()
@click.option("--start-date", type=click.STRING, required=True)
@click.option("--end-date", type=click.STRING, required=True)
@click.option("--download-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option(
    "--geckodriver-path", type=click.Path(path_type=pathlib.Path), required=True
)
@click.option("--head", type=click.INT, default=None)
def main(start_date, end_date, download_path, geckodriver_path, head):
    download_path = download_path.expanduser()
    geckodriver_path = geckodriver_path.expanduser()
    base_url = "https://www.courdecassation.fr/recherche-judilibre?"

    data = {
        "search_api_fulltext": "",
        "judilibre_juridiction": "cc",
        "judilibre_chambre%5B0%5D": "cr",
        "op": "Rechercher%20sur%20judilibre",
    }

    grid = pd.date_range(start_date, end_date, freq="1D")
    failed = []
    for date in grid:
        fs = process_date(date, data, base_url, download_path, geckodriver_path, head)
        failed += fs
    logger.info(f"download complete, {len(failed)} failures")
    logger.info(f"failed ids: {failed}")


if __name__ == "__main__":
    logger_conf = "logging.conf"
    logging.config.fileConfig(logger_conf, disable_existing_loggers=False)
    main()
