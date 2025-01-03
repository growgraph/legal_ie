import logging
import sys

import pathlib
import click

from legal_ie.lake import crawl_directories, process_file

logger = logging.getLogger(__name__)


def process_batch(batch, fresh):
    ts = [(rfp, *process_file(fp), fresh) for fp, rfp in batch]
    template = "Chambre criminelle"
    ts_new = [
        (rfp, text, pages, fresh)
        for rfp, text, pages, fresh in ts
        if (pages and template in pages[0]) or (not pages)
    ]
    return ts_new


@click.command()
@click.option("--input-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--head", type=click.INT, default=None)
@click.option("--batch-size", type=click.INT, default=10)
@click.option("--fresh", type=click.BOOL, is_flag=True, default=False)
def main(input_path, head, batch_size, fresh):
    input_path = input_path.expanduser()

    nabs_parts = len(input_path.expanduser().parts)

    files = sorted(crawl_directories(input_path.expanduser()))
    files = [(fp, pathlib.Path(*fp.parts[nabs_parts:])) for fp in files]
    if head is not None:
        files = files[:head]

    batches = [
        files[i * batch_size : (i + 1) * batch_size]
        for i in range(int(len(files) / batch_size) + 1)
    ]

    total_items = []
    for batch in batches:
        items = process_batch(batch, fresh)
        print(len(items))
        total_items += items

    print(len(total_items))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()
