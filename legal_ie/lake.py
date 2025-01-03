"""
the lake is currently represented by file system
"""

import json
import pathlib

import fitz
from legal_ie.pdf import chop_footer
import logging
import logging.config

logger = logging.getLogger(__name__)


def crawl_directories(input_path: pathlib.Path, suffixes=(".pdf", ".json")):
    file_paths = []

    if not input_path.is_dir():
        print(f"The path {input_path} is not a valid directory.")
        return file_paths

    for file in input_path.rglob("*"):
        if file.is_file() and file.suffix in suffixes:
            file_paths.append(file)
    return file_paths


def process_file(fp_abs):
    if fp_abs.suffix == ".pdf":
        try:
            pdf_document = fitz.open(fp_abs)
        except Exception:
            logger.error(f" failed to open {fp_abs}")
            return "", []
        pages = []
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            raw_text = page.get_text()
            chopped_text = chop_footer(raw_text)
            pages += [chopped_text]
        text = join_pages(pages)
    elif fp_abs.suffix == ".json":
        with open(fp_abs) as f:
            jdata = json.load(f)

        text = jdata["text"]
        pages = []
    else:
        text = ""
        pages = []
    return text, pages


def join_pages(pages: list[str]):
    full_text = []
    for pt in pages:
        paragraph = []
        lines = pt.splitlines()
        for line in lines:
            if line.strip():
                if paragraph and not paragraph[-1].strip().endswith(
                    (".", "!", "?", '"', "__")
                ):
                    paragraph[-1] += f" {line.strip()}"
                else:
                    paragraph.append(line.strip())

        full_text.extend(paragraph)
    return "\n".join(full_text)
