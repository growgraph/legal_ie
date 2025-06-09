from time import sleep
from urllib.parse import urlencode

from selenium import webdriver
from selenium.webdriver.firefox.service import Service

from tqdm import tqdm
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
import time
import re
import pathlib
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import logging

logger = logging.getLogger(__name__)


def download_pdf(driver, pdf_url, download_path: pathlib.Path, sleep_time=3):
    driver.execute_script("window.open(arguments[0], '_blank');", pdf_url)
    sleep(sleep_time)
    wait_for_pdf_download(driver, download_path, timeout=5)


def wait_for_pdf_download(driver, download_path: pathlib.Path, timeout):
    incomplete_extension = ".part"  # For Firefox's incomplete downloads

    def is_download_complete(driver):
        download_files = [file for file in download_path.iterdir() if file.is_file()]
        incomplete_files = [
            file for file in download_files if file.suffix == incomplete_extension
        ]
        return False if incomplete_files else True

    WebDriverWait(driver, timeout).until(is_download_complete)
    return


def fetch_decision_links(driver, base_url, head=None, sleep_time=1):
    """Fetches decision links from all pages."""
    total_pages = fetch_total_pages(base_url, driver)

    all_links = []

    for page in range(total_pages):
        # Construct the URL for each page
        page_url = f"{base_url}&page={page}"

        driver.get(page_url)
        time.sleep(sleep_time)

        # Extract links to decisions
        decision_elements = driver.find_elements(
            By.XPATH, "//a[contains(@href, '/decision/')]"
        )
        links = [elem.get_attribute("href") for elem in decision_elements]

        all_links += links
        if head is not None and page >= head - 1:
            break

    return all_links


def fetch_total_pages(base_url, driver, fetch_timeout=10):
    """Fetches the total number of pages from the first page."""
    driver.get(base_url)

    try:
        wait = WebDriverWait(driver, fetch_timeout)
        result_element = wait.until(
            EC.presence_of_element_located((By.XPATH, "//h2[@class='h6-like']"))
        )
        result_text = result_element.text
    except TimeoutException:
        logger.error("Timeout: Could not locate the total results text.")
        return 0
    except NoSuchElementException:
        logger.error("NoSuchElement: Could not find the total results element.")
        return 0

    # Match something like "118 résultat(s) - 12 page(s)"
    match = re.search(r"(\d+) résultat\(s\)\s*-\s*(\d+) page\(s\)", result_text)
    if match:
        total_results = int(match.group(1))
        total_pages = int(match.group(2))
        logger.info(f"Found {total_results} results across {total_pages} pages.")
        return total_pages
    return 0


def download_pdfs(driver, decision_links, download_path, without_zonage=True):
    failed = []

    pdf_kind = 0 if without_zonage else 1

    for link in tqdm(decision_links):
        doc_id = link.split("/")[-1].split("?")[0]
        pdf_url = f"https://www.courdecassation.fr/decision/export/{doc_id}/{pdf_kind}"
        try:
            download_pdf(driver, pdf_url, download_path)
        except Exception as e:
            failed += [doc_id]
            logger.error(f"Error downloading PDF for {link}: {e}")

    driver.quit()
    return failed


def init_ff_options(download_path):
    # Setup Selenium WebDriver for Firefox
    firefox_options = Options()
    firefox_options.add_argument("--headless")
    firefox_options.add_argument("--no-sandbox")
    firefox_options.add_argument("--disable-gpu")
    firefox_options.set_preference("browser.download.folderList", 2)
    firefox_options.set_preference("browser.download.manager.showWhenStarting", False)
    firefox_options.set_preference("browser.download.dir", download_path.as_posix())
    firefox_options.set_preference(
        "browser.helperApps.neverAsk.saveToDisk", "application/pdf"
    )
    firefox_options.set_preference("browser.download.manager.closeWhenDone", True)
    firefox_options.set_preference(
        "browser.safebrowsing.downloads.enabled", False
    )  # Disable safe browsing for downloads
    firefox_options.set_preference(
        "dom.popup_allowed_events", "change click dblclick mouseup"
    )  # Allow pop-ups
    firefox_options.set_preference(
        "dom.disable_open_during_load", False
    )  # Allow opening during load
    firefox_options.set_preference(
        "browser.helperApps.alwaysAsk.force", False
    )  # No "open with" dialog
    firefox_options.set_preference(
        "security.dialog_enable_delay", 0
    )  # Disable dialog delays
    firefox_options.set_preference("security.warn_entering_secure", False)
    firefox_options.set_preference("security.warn_leaving_secure", False)
    firefox_options.set_preference("security.warn_viewing_mixed", False)
    return firefox_options


def process_date(
    date, data, base_url, download_path, geckodriver_path, head, geckodriver_port=4444
):
    date_str = date.date().isoformat()
    logger.info(f"Processing {date_str}")
    data["date_du"] = date_str
    data["date_au"] = date_str

    download_subpath = download_path / date_str

    if not download_subpath.exists():
        download_subpath.mkdir(parents=True, exist_ok=True)

    ff_options = init_ff_options(download_subpath)
    service = Service(geckodriver_path.as_posix(), port=geckodriver_port)
    driver = webdriver.Firefox(service=service, options=ff_options)
    query_string = urlencode(data)
    full_url = f"{base_url}{query_string}"
    decision_links = fetch_decision_links(driver, full_url, head=head)

    logger.info(f"{date.date().isoformat()} : found {len(decision_links)} decisions")

    failed_ids = download_pdfs(driver, decision_links, download_subpath)
    logger.info(
        f"{date.date().isoformat()} : download complete, {len(failed_ids)} failures"
    )
    return failed_ids
