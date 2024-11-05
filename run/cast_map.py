import logging
import sys

import pathlib
import click
import os

logger = logging.getLogger(__name__)


@click.command()
@click.option("--input-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--onto-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--env-path", type=click.Path(path_type=pathlib.Path), required=True)
def main(input_path, onto_path, env_path):
    jsons = [
        f
        for f in input_path.expanduser().iterdir()
        if f.suffix == ".json" and f.is_file()
    ]
    csvs = [f for f in input_path.expanduser().iterdir() if f.suffix == ".csv"]
    csv_stems = [x.stem for x in csvs]
    json_proc = [jf for jf in jsons if jf.stem not in csv_stems]
    print(f"Will be processing {len(json_proc)} docs")

    for jf in sorted(json_proc):
        cli_call = f"python -m run.cast_triple --onto-path {onto_path.as_posix()} --env-path {env_path.as_posix()} --input-path {jf.as_posix()}"
        os.system(cli_call)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()
