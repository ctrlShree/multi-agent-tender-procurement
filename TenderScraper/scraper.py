import os
import re
import sys
import json
import time
import shutil
import zipfile
import argparse
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager
from groq import Groq
from dotenv import load_dotenv

# Load API key from the env file sitting next to this script
load_dotenv(dotenv_path=Path(__file__).parent / "env")

# Configuration
MAX_TENDERS = 25
MAX_PER_ORG = 50

# Groq API key, read from env file or environment variable
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

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


# LLM-based category mapping

def get_product_categories(driver):
    """
    Navigates to the Tender Search By Organisation page and reads all
    available options from the Product Category dropdown.
    Returns a list of option strings (excluding the default '-Select-').
    """
    SEARCH_URL = "https://etenders.gov.in/eprocure/app?page=FrontEndTendersByOrganisation&service=page"
    driver.get(SEARCH_URL)
    time.sleep(ORG_PAGE_WAIT)

    try:
        select_el = driver.find_element(By.XPATH, "//select[contains(@id,'prodCategory') or contains(@name,'prodCategory') or contains(@id,'productCategory')]")
    except Exception:
        # Fallback: find all <select> elements and pick the one labelled Product Category
        selects = driver.find_elements(By.TAG_NAME, "select")
        select_el = None
        for s in selects:
            # Look for the label near this select
            try:
                parent_text = s.find_element(By.XPATH, "ancestor::tr[1]").text
                if "Product Category" in parent_text:
                    select_el = s
                    break
            except Exception:
                continue

    if not select_el:
        print("  Warning: Could not find Product Category dropdown. Scraping without filter.")
        return []

    select = Select(select_el)
    options = [opt.text.strip() for opt in select.options if opt.text.strip() and opt.text.strip() != "-Select-"]
    print(f"  Found {len(options)} product categories in dropdown.")
    return options


def map_profile_to_categories(company_profile, available_categories, groq_api_key):
    """
    Uses Groq/Llama to map a free-text company profile to the most relevant
    Product Category options available in the portal dropdown.

    Returns a list of matched category strings (1 or more).
    Falls back to empty list if the API call fails.
    """
    if not groq_api_key:
        print("  Warning: No GROQ_API_KEY set. Cannot map profile to category.")
        return []

    if not available_categories:
        return []

    categories_str = "\n".join(f"- {c}" for c in available_categories)

    prompt = f"""You are helping a procurement scraper select the most relevant product categories from a government tender portal.

Company profile: "{company_profile}"

Available product categories on the portal:
{categories_str}

Instructions:
- Select ALL categories that are relevant to the company profile.
- Return ONLY a JSON array of the exact category strings from the list above, nothing else.
- If nothing matches well, return the single closest match.
- Example output: ["Electrical Works", "Civil Works"]

Return only the JSON array, no explanation."""

    try:
        client = Groq(api_key=groq_api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        # Parse the JSON array from the response
        matched = json.loads(raw)
        # Validate — only keep options that actually exist in the dropdown
        matched = [c for c in matched if c in available_categories]
        print(f"  LLM mapped profile to: {matched}")
        return matched
    except Exception as e:
        print(f"  Warning: LLM category mapping failed ({e}). Scraping without filter.")
        return []


def search_by_category(driver, category, captcha_solved):
    """
    On the Tender Search By Organisation page, selects the given Product Category,
    enters the CAPTCHA (manual on first run), and clicks Search.
    Returns True if the search results loaded successfully.
    """
    SEARCH_URL = "https://etenders.gov.in/eprocure/app?page=FrontEndTendersByOrganisation&service=page"
    driver.get(SEARCH_URL)
    time.sleep(ORG_PAGE_WAIT)

    # Select the Product Category
    try:
        select_els = driver.find_elements(By.TAG_NAME, "select")
        product_select = None
        for s in select_els:
            try:
                parent_text = s.find_element(By.XPATH, "ancestor::tr[1]").text
                if "Product Category" in parent_text:
                    product_select = s
                    break
            except Exception:
                continue

        if product_select:
            Select(product_select).select_by_visible_text(category)
            print(f"  Selected Product Category: '{category}'")
            time.sleep(1)
        else:
            print(f"  Warning: Could not select Product Category '{category}'")
    except Exception as e:
        print(f"  Warning: Dropdown selection failed: {e}")

    # Handle CAPTCHA
    if not captcha_solved:
        print("\n  Please solve the CAPTCHA in the browser window.")
        captcha_text = input("  Type the CAPTCHA text shown and press ENTER: ").strip()
        try:
            captcha_input = driver.find_element(By.XPATH, "//input[contains(@id,'captcha') or contains(@name,'captcha') or contains(@placeholder,'aptcha')]")
            captcha_input.clear()
            captcha_input.send_keys(captcha_text)
        except Exception:
            print("  Warning: Could not find CAPTCHA input field. Please fill it manually.")
            input("  Press ENTER once you have filled the CAPTCHA and are ready to search: ")

    # Click Search
    try:
        search_btn = driver.find_element(By.XPATH, "//input[@value='Search'] | //button[contains(text(),'Search')]")
        driver.execute_script("arguments[0].click();", search_btn)
        time.sleep(ORG_PAGE_WAIT)
        print(f"  Search submitted for category: '{category}'")
        return True
    except Exception as e:
        print(f"  Warning: Could not click Search button: {e}")
        return False


# Get organisation links

def get_org_links(driver):
    print("Loading organisation page...")
    driver.get("https://etenders.gov.in/eprocure/app?page=FrontEndTendersByOrganisation&service=page")
    time.sleep(ORG_PAGE_WAIT)
    print("Page loaded.")

    all_links = driver.find_elements(By.TAG_NAME, "a")
    orgs = []
    seen = set()

    for link in all_links:
        href = link.get_attribute("href") or ""
        text = link.text.strip()
        if ("DirectLink" in href
            and "FrontEndTendersByOrganisation" in href
            and text.isdigit()
            and href not in seen):
            count = int(text)
            if 0 < count <= MAX_PER_ORG:
                seen.add(href)
                orgs.append({"href": href, "count": count})

    print(f"Found {len(orgs)} organisations (≤{MAX_PER_ORG} tenders each).\n")
    return orgs


def get_org_links_from_current_page(driver):
    """Reads org links from the already-loaded search results page."""
    all_links = driver.find_elements(By.TAG_NAME, "a")
    orgs = []
    seen = set()

    for link in all_links:
        href = link.get_attribute("href") or ""
        text = link.text.strip()
        if ("DirectLink" in href
            and "FrontEndTendersByOrganisation" in href
            and text.isdigit()
            and href not in seen):
            count = int(text)
            if 0 < count <= MAX_PER_ORG:
                seen.add(href)
                orgs.append({"href": href, "count": count})

    print(f"Found {len(orgs)} organisations in filtered results (≤{MAX_PER_ORG} tenders each).\n")
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

def scrape_category(driver, category, stats, total_so_far, captcha_done):
    """
    Runs the full scrape pipeline for a single product category:
    1. Submits the filtered search for this category
    2. Reads org links from results
    3. Iterates orgs → tenders → download

    Returns (new_total, captcha_done)
    """
    total = total_so_far

    print(f"\n{'━' * 58}")
    print(f"  Searching category: '{category}'")
    print(f"{'━' * 58}")

    search_ok = search_by_category(driver, category, captcha_solved=captcha_done)
    if not search_ok:
        print(f"  Skipping category '{category}' — search failed.")
        return total, captcha_done

    # First search requires CAPTCHA, mark as done after first successful search
    captcha_done = True

    orgs = get_org_links_from_current_page(driver)
    if not orgs:
        print(f"  No organisations found for category '{category}'.")
        return total, captcha_done

    for org_idx, org in enumerate(orgs):
        if total >= MAX_TENDERS:
            break

        print(f"{'━' * 58}")
        print(f"  Org {org_idx+1}/{len(orgs)} — {org['count']} tenders  [{category}]")
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
                metadata["matched_category"] = category   # record which category matched
                if metadata.get("tender_reference_number"):
                    print(f"    Ref: {metadata['tender_reference_number']}")

                zip_files = download_zip(driver, captcha_done=True)

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

    return total, captcha_done


def main():
    parser = argparse.ArgumentParser(description="eTenders scraper — category filtered by company profile")
    parser.add_argument("--profile", default=None, help="Company capability description (e.g. 'electrical works, conveyors')")
    args = parser.parse_args()

    company_profile = args.profile
    if not company_profile:
        company_profile = input("Enter your company profile / capabilities: ").strip()
    if not company_profile:
        print("Error: company profile is required.")
        sys.exit(1)

    print("=" * 58)
    print("  eTenders.gov.in — ZIP Scraper v7 (Category Filtered)")
    print("=" * 58)
    print(f"  Profile: {company_profile}")
    print(f"  Target:  {MAX_TENDERS} tenders")
    print(f"  Output:  {EXTRACT_DIR}\n")

    driver = get_driver()
    total = 0
    stats = {"orgs": 0, "ok": 0, "files": 0, "failed": 0}
    captcha_done = False
    matched_categories = []  # populated after LLM mapping

    try:
        # Step 1: Read available product categories from the portal dropdown
        print("Step 1: Reading available Product Categories from portal...")
        available_categories = get_product_categories(driver)

        # Step 2: Use LLM to map company profile to relevant categories
        if available_categories and GROQ_API_KEY:
            print("Step 2: Mapping company profile to portal categories via LLM...")
            matched_categories = map_profile_to_categories(
                company_profile, available_categories, GROQ_API_KEY
            )
        else:
            matched_categories = []

        # Step 3: If no categories matched, fall back to unfiltered scrape
        if not matched_categories:
            print("  No category match found — falling back to unfiltered scrape.\n")
            orgs = get_org_links(driver)
            if not orgs:
                print("No organisations found!")
                return
            # Run unfiltered scrape using original logic
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
                    print(f"  [{total}/{MAX_TENDERS}] {short_title}")
                    try:
                        driver.get(tender["href"])
                        time.sleep(TENDER_PAGE_WAIT)
                        metadata = get_metadata(driver)
                        metadata["title"] = tender["title"]
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
        else:
            # Step 4: Scrape each matched category in turn
            print(f"\nStep 3: Scraping {len(matched_categories)} matched categories...\n")
            for category in matched_categories:
                if total >= MAX_TENDERS:
                    break
                total, captcha_done = scrape_category(
                    driver, category, stats, total, captcha_done
                )

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
    print(f"  Profile:          {company_profile}")
    cats_display = ", ".join(matched_categories) if matched_categories else "unfiltered"
    print(f"  Categories used:  {cats_display}")
    print(f"  Orgs scraped:     {stats['orgs']}")
    print(f"  Tenders total:    {total}")
    print(f"  ZIPs downloaded:  {stats['ok']}")
    print(f"  Files extracted:  {stats['files']}")
    print(f"  Failed/no ZIP:    {stats['failed']}")
    print(f"  Output:           {EXTRACT_DIR}")
    print(f"{'=' * 58}")


if __name__ == "__main__":
    main()