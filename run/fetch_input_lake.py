"""
scrape appeal pdfs from www.courdecassation.fr


"""

import pandas as pd
import pathlib
import click
import logging.config
import suthing

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
        "judilibre_chambre": "cr",
    }

    grid = pd.date_range(start_date, end_date, freq="1D")
    failed = []
    for date in grid:
        fs = process_date(date, data, base_url, download_path, geckodriver_path, head)
        failed += fs
    logger.info(f"download complete, {len(failed)} failures")
    logger.info(f"failed ids: {failed}")
    suthing.FileHandle.dump(failed, download_path / "failed.json")


if __name__ == "__main__":
    logger_conf = "logging.conf"
    logging.config.fileConfig(logger_conf, disable_existing_loggers=False)
    main()
