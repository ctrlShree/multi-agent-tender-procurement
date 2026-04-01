import os
import re
import json
import time
import shutil
import zipfile
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# Configuration
MAX_TENDERS = 5         
MAX_PER_ORG = 50           

# Output folders
BASE_DIR = os.path.join(os.getcwd(), "tender_data")
ZIP_DIR = os.path.join(BASE_DIR, "zips")
EXTRACT_DIR = os.path.join(BASE_DIR, "extracted")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")

# Timing
ORG_PAGE_WAIT = 15         # Org listing page is slow
TENDER_LIST_WAIT = 10      # After clicking org
TENDER_PAGE_WAIT = 8       # After opening a tender
BETWEEN_TENDERS = 2        # Pause between tenders
ZIP_TIMEOUT = 120          # Max wait for ZIP download

for d in [BASE_DIR, ZIP_DIR, EXTRACT_DIR, METADATA_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)


# Browser setup

def get_driver():
    opts = Options()
    prefs = {
        "download.default_directory": ZIP_DIR,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True,
        "safebrowsing.enabled": True,
        "safebrowsing.disable_download_protection": True,
    }
    opts.add_experimental_option("prefs", prefs)
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")

    svc = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=svc, options=opts)
    driver.set_page_load_timeout(120)
    return driver


# Helper functions

def safe_name(text, max_len=60):
    return re.sub(r'[\\/*?:"<>|\s]+', "_", text).strip("_")[:max_len]


def files_in(d):
    return set(os.listdir(d))


def wait_for_zip(before_files, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        current = files_in(ZIP_DIR)
        new = current - before_files
        downloading = [f for f in new if f.endswith((".crdownload", ".tmp"))]
        if new and not downloading:
            return list(new)
        time.sleep(1)
    return []


def extract_zip(zip_path, dest):
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
            return zf.namelist()
    except zipfile.BadZipFile:
        shutil.copy2(zip_path, dest)
        return [os.path.basename(zip_path)]
    except Exception as e:
        print(f"      Extract error: {e}")
        return []


# Get organisation links

def get_org_links(driver):
    print("Loading organisation page...")
    driver.get("https://etenders.gov.in/eprocure/app?page=FrontEndTendersByOrganisation&service=page")
    time.sleep(ORG_PAGE_WAIT)
    print("Page loaded.")

    all_links = driver.find_elements(By.TAG_NAME, "a")
    orgs = []

    for link in all_links:
        href = link.get_attribute("href") or ""
        text = link.text.strip()
        if ("DirectLink" in href
            and "FrontEndTendersByOrganisation" in href
            and text.isdigit()):
            count = int(text)
            if 0 < count <= MAX_PER_ORG:
                orgs.append({"href": href, "count": count})

    print(f"Found {len(orgs)} organisations (≤{MAX_PER_ORG} tenders each).\n")
    return orgs


# Fetch tender links

def get_tender_links(driver):
  
    all_links = driver.find_elements(By.TAG_NAME, "a")
    tenders = []

    for link in all_links:
        href = link.get_attribute("href") or ""
        text = link.text.strip()
        if ("DirectLink" in href
            and text
            and len(text) > 5
            and not text.isdigit()
            and "FrontEndTendersByOrganisation" not in href):
            tenders.append({"href": href, "title": text})

    return tenders


# Download tender as zip

def download_zip(driver, captcha_done=False):

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

    # Find the ZIP link
    try:
        zip_links = driver.find_elements(By.PARTIAL_LINK_TEXT, "Download as zip")
        if not zip_links:
            zip_links = driver.find_elements(By.PARTIAL_LINK_TEXT, "download as zip")
        if not zip_links:
            zip_links = driver.find_elements(By.PARTIAL_LINK_TEXT, "zip file")
    except Exception:
        zip_links = []

    if not zip_links:
        print("    ✗ No 'Download as zip' link found")
        return []

    # Click the zip link
    zip_link = zip_links[0]
    print(f"    Clicking 'Download as zip file'...")

    before = files_in(ZIP_DIR)

    try:
        driver.execute_script("arguments[0].click();", zip_link)
    except Exception:
        zip_href = zip_link.get_attribute("href") or ""
        if zip_href.startswith("http"):
            driver.get(zip_href)

    if not captcha_done:
        time.sleep(3)
        print( " If a CAPTCHA appeared, please solve it in the browser.")
        input("  Press ENTER after solving the CAPTCHA...")
        print("  CAPTCHA done! Re-clicking ZIP download...\n")

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        before = files_in(ZIP_DIR)

        try:
            zip_links2 = driver.find_elements(By.PARTIAL_LINK_TEXT, "Download as zip")
            if not zip_links2:
                zip_links2 = driver.find_elements(By.PARTIAL_LINK_TEXT, "zip file")
            if zip_links2:
                driver.execute_script("arguments[0].click();", zip_links2[0])
        except Exception:
            pass

    time.sleep(3)

    new_files = wait_for_zip(before, ZIP_TIMEOUT)
    downloaded = [f for f in new_files if not f.endswith((".crdownload", ".tmp"))]

    for f in downloaded:
        size = os.path.getsize(os.path.join(ZIP_DIR, f)) / 1024
        print(f"    ✓ Downloaded: {f} ({size:.1f} KB)")

    return downloaded


# Collect metadata from tender page

def get_metadata(driver):
   
    meta = {}

    important_fields = [
        "Tender Reference Number",
        "Tender ID",
        "Organisation Chain",
        "Tender Category",
        "Tender Type",
        "Form Of Contract",
        "No. of Covers",
        "Tender Value in ₹",
        "Tender Value",
        "EMD Amount in ₹",
        "EMD Amount",
        "Tender Fee in ₹",
        "Tender Fee",
        "Product Category",
        "Work Description",
        "Pre Qualification",
        "Document Download / Sale Start Date",
        "Document Download / Sale End Date",
        "Clarification Start Date",
        "Clarification End Date",
        "Bid Submission Start Date",
        "Bid Submission End Date",
        "Bid Opening Date",
        "Bid Validity(Days)",
        "ItemWise Technical Evaluation Allowed",
        "Allow Two Stage Bidding",
        "Is Multi Currency Allowed For BOQ",
        "Withdrawal Allowed",
    ]
    for field in important_fields:
        key = field.lower().replace(" ", "_").replace("/", "_").replace("₹", "rs")
        key = re.sub(r'[^a-z0-9_]', '', key)
        try:
            el = driver.find_element(By.XPATH,
                f"//td[contains(text(),'{field}')]/following-sibling::td"
            )
            val = el.text.strip()
            if val:
                meta[key] = val
        except Exception:
            pass

    try:
        body = driver.find_element(By.TAG_NAME, "body")
        meta["_raw_page_text"] = body.text[:5000]
    except Exception:
        pass

    return meta


# Extract ZIP 

def extract_and_save(tender_num, metadata, zip_files):
    ref = (metadata.get("tender_reference_number", "")
           or metadata.get("ref", "")
           or metadata.get("tender_id", "")
           or "tender")
    folder_name = f"{tender_num:03d}_{safe_name(ref)}"
    folder = os.path.join(EXTRACT_DIR, folder_name)
    Path(folder).mkdir(exist_ok=True)

    total = 0
    for fname in zip_files:
        src = os.path.join(ZIP_DIR, fname)
        if not os.path.exists(src):
            continue
        names = extract_zip(src, folder)
        total += len(names)
        if names:
            print(f"    Extracted {len(names)} files:")
            for n in names:
                print(f"      - {n}")

    # Save metadata
    metadata["tender_num"] = tender_num
    metadata["zip_files"] = zip_files
    metadata["extracted_to"] = folder
    meta_path = os.path.join(METADATA_DIR, f"tender_{tender_num:03d}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return total, folder

def main():
    print("=" * 58)
    print("  eTenders.gov.in — ZIP Scraper v6")
    print("=" * 58)
    print(f"  Target:  {MAX_TENDERS} tenders")
    print(f"  Output:  {EXTRACT_DIR}\n")

    driver = get_driver()
    total = 0
    stats = {"orgs": 0, "ok": 0, "files": 0, "failed": 0}
    captcha_done = False

    try:
        orgs = get_org_links(driver)
        if not orgs:
            print("No organisations found!")
            return

        for org_idx, org in enumerate(orgs):
            if total >= MAX_TENDERS:
                break

            print(f"{'━' * 58}")
            print(f"  Org {org_idx+1}/{len(orgs)} — {org['count']} tenders")
            print(f"{'━' * 58}")

            driver.get(org["href"])
            time.sleep(TENDER_LIST_WAIT)

            stats["orgs"] += 1

            tenders = get_tender_links(driver)
            print(f"  Found {len(tenders)} tenders.\n")

            if not tenders:
                continue

            org_url = driver.current_url

            for tender in tenders:
                if total >= MAX_TENDERS:
                    break

                total += 1
                short_title = tender["title"][:50]
                if len(tender["title"]) > 50:
                    short_title += "..."
                print(f"  [{total}/{MAX_TENDERS}] {short_title}")

                try:
                    driver.get(tender["href"])
                    time.sleep(TENDER_PAGE_WAIT)

                    metadata = get_metadata(driver)
                    metadata["title"] = tender["title"]
                    if metadata.get("tender_reference_number"):
                        print(f"    Ref: {metadata['tender_reference_number']}")

                    zip_files = download_zip(driver, captcha_done)
                    if not captcha_done and zip_files:
                        captcha_done = True

                    if zip_files:
                        count, folder = extract_and_save(total, metadata, zip_files)
                        stats["ok"] += 1
                        stats["files"] += count
                        print(f"    ✓ Done — {count} files in {os.path.basename(folder)}/")
                    else:
                        stats["failed"] += 1

                except Exception as e:
                    print(f"    ✗ Error: {str(e)[:80]}")
                    stats["failed"] += 1

                try:
                    driver.get(org_url)
                    time.sleep(5)
                except Exception:
                    pass

                time.sleep(BETWEEN_TENDERS)

            print()

    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C)")

    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        try:
            driver.quit()
            print("\nBrowser closed.")
        except Exception:
            print("\nBrowser already closed.")

    # Conclusion
    print(f"\n{'=' * 58}")
    print("  DONE")
    print(f"{'=' * 58}")
    print(f"  Orgs scraped:     {stats['orgs']}")
    print(f"  Tenders total:    {total}")
    print(f"  ZIPs downloaded:  {stats['ok']}")
    print(f"  Files extracted:  {stats['files']}")
    print(f"  Failed/no ZIP:    {stats['failed']}")
    print(f"  Output:           {EXTRACT_DIR}")
    print(f"{'=' * 58}")


if __name__ == "__main__":
    main()