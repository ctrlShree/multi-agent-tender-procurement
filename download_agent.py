import os
import json
import time
import zipfile
import requests
from pathlib import Path
from dataclasses import dataclass, asdict

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager


@dataclass
class TenderMetadata:
    """Structured metadata extracted from a tender page."""
    tender_id: str
    reference_number: str
    organisation: str
    title: str
    category: str
    tender_value: str
    emd_amount: str
    published_date: str
    submission_deadline: str
    opening_date: str
    source_url: str
    download_path: str
    files_extracted: list
    download_success: bool


class DownloadAgent:
    """
    Agent responsible for:
    1. Navigating eTenders.gov.in to find relevant tenders
    2. Applying a keyword relevance pre-filter against the company profile
    3. Downloading tender ZIP files and extracting their contents
    4. Returning structured metadata for each downloaded tender
    """

    BASE_URL = "https://etenders.gov.in/eprocure/app"
    ORG_LISTING_URL = f"{BASE_URL}?page=FrontEndOrganisationList"

    FIELD_LABELS = {
        "reference_number":    ["Tender Reference Number", "Ref No"],
        "organisation":        ["Organisation Name", "Organisation Chain"],
        "title":               ["Tender Title", "Work Description"],
        "category":            ["Product Category", "Tender Category"],
        "tender_value":        ["Tender Value (INR)", "Estimated Value"],
        "emd_amount":          ["EMD Amount (INR)", "Earnest Money Deposit"],
        "published_date":      ["Published Date", "Bid Document Download Start Date"],
        "submission_deadline": ["Bid Submission End Date", "Submission Date"],
        "opening_date":        ["Bid Opening Date"],
    }

    def __init__(self, download_dir, company_profile, max_orgs=50, max_tenders_per_org=5):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.keywords = [kw.strip().lower() for kw in company_profile.split(",")]
        self.max_orgs = max_orgs
        self.max_tenders_per_org = max_tenders_per_org
        self.driver = None
        self.wait = None
        self.results = []

    def start_session(self):
        """
        Launches a Chrome browser session with download preferences configured.
        Navigates to the portal and pauses for manual CAPTCHA resolution.
        All subsequent navigation runs automatically within the same live session.
        """
        opts = Options()
        opts.add_experimental_option("prefs", {
            "download.default_directory": str(self.download_dir.resolve()),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "safebrowsing.disable_download_protection": True,
            "plugins.always_open_pdf_externally": True,
        })
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")

        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=opts)
        self.driver.set_page_load_timeout(120)
        self.wait = WebDriverWait(self.driver, 20)

        print("Opening eTenders portal...")
        self.driver.get(self.ORG_LISTING_URL)
        time.sleep(5)

        print("\nIf a CAPTCHA is shown, please solve it in the browser window.")
        input("Press ENTER once the page has fully loaded and any CAPTCHA is solved: ")
        print("Session ready. Starting automated navigation.\n")

    def close(self):
        """Closes the browser session cleanly."""
        if self.driver:
            self.driver.quit()

    def run(self, max_tenders=150):
        """
        Main pipeline entry point. Coordinates the full download workflow:
        1. Retrieve list of organisation URLs from the portal
        2. For each organisation, retrieve individual tender page URLs
        3. For each tender, run relevance check, scrape metadata, and download

        Returns:
            List of TenderMetadata objects for all processed tenders
        """
        if not self.driver:
            raise RuntimeError("Call start_session() before run()")

        org_links = self._get_organisation_links()
        print(f"Found {len(org_links)} organisations. Processing up to {self.max_orgs}...")

        total_downloaded = 0

        for org_url in org_links[:self.max_orgs]:
            if total_downloaded >= max_tenders:
                print(f"Reached target of {max_tenders} tenders. Stopping.")
                break

            tender_links = self._get_tender_links_for_org(org_url)

            for tender_url in tender_links[:self.max_tenders_per_org]:
                if total_downloaded >= max_tenders:
                    break

                metadata = self._process_tender(tender_url)
                if metadata:
                    self.results.append(metadata)
                    if metadata.download_success:
                        total_downloaded += 1
                        print(f"  [{total_downloaded}/{max_tenders}] {metadata.title[:60]}...")

        self._save_metadata_index()
        print(f"\nDone. {total_downloaded} tenders saved to {self.download_dir}")
        return self.results

    # -------------------------------------------------------------------------

    def _get_organisation_links(self):
        """
        Retrieves organisation page URLs from the portal listing page.
        Filters anchor tags by href pattern to avoid XPath-induced browser freezes.

        Returns:
            Deduplicated list of organisation page URLs
        """
        try:
            self.driver.get(self.ORG_LISTING_URL)
            time.sleep(8)
        except Exception as e:
            print(f"Warning: page load issue on org listing: {e}")

        all_links = self.driver.find_elements(By.TAG_NAME, "a")
        seen = set()
        org_links = []

        for link in all_links:
            href = link.get_attribute("href") or ""
            if "FrontEndTendersByOrganisation" in href and "DirectLink" in href and href not in seen:
                # Skip orgs with very high tender counts — those listing pages are
                # extremely slow and are more likely to be irrelevant bulk categories
                text = link.text.strip()
                if text.isdigit() and int(text) > 50:
                    continue
                seen.add(href)
                org_links.append(href)

        return org_links

    def _get_tender_links_for_org(self, org_url):
        """
        Navigates to a single organisation's tender listing page and
        returns all individual tender detail page URLs found on it.

        Returns:
            Deduplicated list of tender detail page URLs
        """
        try:
            self.driver.get(org_url)
            time.sleep(8)
        except Exception as e:
            print(f"Warning: page load issue navigating to org: {e}")
            return []

        all_links = self.driver.find_elements(By.TAG_NAME, "a")
        seen = set()
        tender_links = []

        for link in all_links:
            href = link.get_attribute("href") or ""
            if "FrontEndTenderDetails" in href and href not in seen:
                seen.add(href)
                tender_links.append(href)

        return tender_links

    def _process_tender(self, tender_url):
        """
        Handles a single tender page end-to-end:
        1. Navigates to the tender detail page
        2. Scrapes structured metadata fields
        3. Applies keyword relevance filter — skips if not relevant
        4. Downloads and extracts the ZIP file
        5. Writes a per-tender metadata.json to disk

        Returns:
            Populated TenderMetadata object, or None if tender is irrelevant
        """
        try:
            self.driver.get(tender_url)
            time.sleep(6)
        except Exception as e:
            print(f"  Warning: failed to load tender page: {e}")
            return None

        fields = self._scrape_tender_fields()
        title    = fields.get("title", "")
        category = fields.get("category", "")

        if not self._is_relevant(title, category):
            return None

        # Build a safe folder name from the reference number
        ref = fields.get("reference_number", "unknown")
        safe_ref = "".join(c if c.isalnum() else "_" for c in ref)[:60]
        tender_folder = self.download_dir / safe_ref
        tender_folder.mkdir(parents=True, exist_ok=True)

        files_extracted, success = self._download_zip(tender_folder)

        # Capture raw page text for the Document Processing Agent to use if
        # PDF extraction fails or is not available
        try:
            raw_text = self.driver.find_element(By.TAG_NAME, "body").text[:8000]
        except Exception:
            raw_text = ""

        metadata = TenderMetadata(
            tender_id=safe_ref,
            reference_number=ref,
            organisation=fields.get("organisation", ""),
            title=title,
            category=category,
            tender_value=fields.get("tender_value", ""),
            emd_amount=fields.get("emd_amount", ""),
            published_date=fields.get("published_date", ""),
            submission_deadline=fields.get("submission_deadline", ""),
            opening_date=fields.get("opening_date", ""),
            source_url=tender_url,
            download_path=str(tender_folder),
            files_extracted=files_extracted,
            download_success=success,
        )

        # Write per-tender metadata JSON (includes raw_text for processing agent)
        meta_dict = asdict(metadata)
        meta_dict["raw_text"] = raw_text
        with open(tender_folder / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2, ensure_ascii=False)

        return metadata

    def _scrape_tender_fields(self):
        """
        Reads the structured data table on a tender detail page by matching
        cell label text against FIELD_LABELS and extracting adjacent values.

        Returns:
            Dictionary of field_name -> extracted value strings
        """
        fields = {key: "" for key in self.FIELD_LABELS}

        try:
            rows = self.driver.find_elements(By.TAG_NAME, "tr")
        except Exception:
            return fields

        for row in rows:
            try:
                cells = row.find_elements(By.TAG_NAME, "td")
            except Exception:
                continue
            if len(cells) < 2:
                continue

            label_text = cells[0].text.strip()
            value_text = cells[1].text.strip()

            for field_name, possible_labels in self.FIELD_LABELS.items():
                for possible_label in possible_labels:
                    if possible_label.lower() in label_text.lower():
                        if value_text:
                            fields[field_name] = value_text
                        break

        return fields

    def _is_relevant(self, title, category):
        """
        Pre-filters tenders by checking whether any company profile keyword
        appears in the tender title or category. Case-insensitive.

        Serves as a lightweight gate before the embedding-based similarity
        scoring performed by the Analysis Agent downstream.

        Returns:
            True if at least one keyword matches, False otherwise
        """
        combined = (title + " " + category).lower()
        return any(kw in combined for kw in self.keywords)

    def _download_zip(self, target_folder):
        """
        Locates the 'Download as zip file' link on the current tender page,
        downloads the file via requests (passing live session cookies for auth),
        extracts the contents to target_folder, and removes the ZIP archive.

        Returns:
            (list of extracted filenames, success bool)
        """
        # Find the ZIP download link by text fragment
        zip_url = None
        for text_fragment in ["Download as zip", "download as zip", "zip file"]:
            try:
                links = self.driver.find_elements(By.PARTIAL_LINK_TEXT, text_fragment)
                if links:
                    zip_url = links[0].get_attribute("href")
                    break
            except Exception:
                continue

        if not zip_url:
            return [], False

        # Transfer live Selenium cookies to requests so the download is authenticated
        session_cookies = {c["name"]: c["value"] for c in self.driver.get_cookies()}

        try:
            response = requests.get(
                zip_url,
                cookies=session_cookies,
                stream=True,
                timeout=120,
            )
            response.raise_for_status()
        except Exception as e:
            print(f"    ZIP download request failed: {e}")
            return [], False

        zip_path = target_folder / "tender_docs.zip"
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract the archive
        extracted_files = []
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(target_folder)
                extracted_files = zf.namelist()
        except zipfile.BadZipFile:
            # Portal occasionally returns the document directly rather than a ZIP
            print("    Warning: file is not a valid ZIP — keeping as-is")
            return ["tender_docs.zip"], True
        except Exception as e:
            print(f"    ZIP extraction error: {e}")
            return [], False

        # Remove the ZIP to save disk space (contents are already extracted)
        try:
            zip_path.unlink()
        except Exception:
            pass

        return extracted_files, True

    def _save_metadata_index(self):
        """
        Serialises all collected TenderMetadata objects to tender_index.json.
        This is the primary handoff artifact to the Document Processing Agent.
        """
        index_path = self.download_dir / "tender_index.json"
        data = [asdict(r) for r in self.results]
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Metadata index saved: {index_path} ({len(data)} records)")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    agent = DownloadAgent(
        download_dir="./tender_data",
        company_profile="software development, cloud computing, IT infrastructure, cybersecurity",
        max_orgs=30,
        max_tenders_per_org=5,
    )

    agent.start_session()
    results = agent.run(max_tenders=150)
    agent.close()

    successful = [r for r in results if r.download_success]
    print(f"\nSummary: {len(successful)} downloaded, "
          f"{len(results) - len(successful)} skipped or failed")
