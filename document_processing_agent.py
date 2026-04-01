"""
Document Processing Agent
==========================
Reads tender_index.json produced by the Download Agent, extracts structured
entities from each tender's text using a BERT-based NER model, and writes
processed_tenders.json for the Analysis Agent.

Model used by default: dslim/bert-base-NER (pre-trained on CoNLL-2003).
Swap MODEL_PATH for a local fine-tuned checkpoint once training is complete.

Run:
    python document_processing_agent.py --data_dir ./tender_data
    python document_processing_agent.py --data_dir ./tender_data --model ./ner_model_finetuned
"""

import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import pdfplumber
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Swap to a local fine-tuned checkpoint once training is done:
#   MODEL_PATH = "./ner_model_finetuned"
MODEL_PATH = "dslim/bert-base-NER"

# BERT hard limit is 512 tokens; we chunk in ~450-word blocks to stay safe
MAX_CHUNK_WORDS = 450


@dataclass
class ProcessedTender:
    """Structured output produced by the Document Processing Agent."""
    tender_id: str
    reference_number: str
    organisation: str
    title: str
    category: str
    tender_value_raw: str
    tender_value_numeric: Optional[float]
    emd_amount_raw: str
    emd_amount_numeric: Optional[float]
    submission_deadline: str
    scope: str
    eligibility: str
    evaluation_criteria: str
    extracted_entities: list          # raw NER output for inspection/debugging
    source_url: str
    processing_success: bool
    error_message: str = ""


class DocumentProcessingAgent:
    """
    Agent responsible for:
    1. Loading raw tender records from tender_index.json
    2. Extracting text from downloaded PDF files where available
    3. Running BERT NER over the text to identify named entities
    4. Using regex patterns to extract domain-specific fields not well-covered by NER
       (monetary values, dates, reference numbers)
    5. Writing structured ProcessedTender records to processed_tenders.json
    """

    def __init__(self, data_dir, model_path=MODEL_PATH):
        """
        Args:
            data_dir:   Directory containing tender_index.json and per-tender
                        sub-folders with downloaded documents.
            model_path: HuggingFace model ID or path to a local fine-tuned model.
        """
        self.data_dir = Path(data_dir)
        self.model_path = model_path
        self.ner_pipeline = None
        self.results = []

    def load_model(self):
        """
        Loads the NER pipeline from HuggingFace or a local fine-tuned checkpoint.
        aggregation_strategy='simple' merges sub-word tokens into full entity spans.
        """
        logger.info(f"Loading NER model: {self.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        self.ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
        )
        logger.info("NER model loaded.")

    def run(self):
        """
        Processes all tenders in tender_index.json.

        Returns:
            List of ProcessedTender objects.
        """
        if self.ner_pipeline is None:
            self.load_model()

        index_path = self.data_dir / "tender_index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"tender_index.json not found in {self.data_dir}. "
                "Run the Download Agent first."
            )

        with open(index_path, "r", encoding="utf-8") as f:
            raw_tenders = json.load(f)

        logger.info(f"Processing {len(raw_tenders)} tenders...")

        for raw in raw_tenders:
            result = self._process_one(raw)
            self.results.append(result)
            status = "OK" if result.processing_success else f"FAILED — {result.error_message}"
            logger.info(f"  [{result.tender_id}] {status}")

        self._save_output()
        logger.info(f"Done. Wrote {len(self.results)} records to processed_tenders.json")
        return self.results

    # -------------------------------------------------------------------------

    def _process_one(self, raw):
        """Processes a single raw tender dict into a ProcessedTender."""
        tender_id = raw.get("tender_id", "UNKNOWN")
        try:
            text = self._get_text(raw)
            if not text:
                raise ValueError("No text available (no raw_text field and no PDF found)")

            entities = self._run_ner(text)
            scope        = self._extract_section(text, ["SCOPE OF WORK", "WORK DESCRIPTION", "SCOPE"])
            eligibility  = self._extract_section(text, ["ELIGIBILITY CRITERIA", "ELIGIBILITY", "PRE QUALIFICATION"])
            eval_section = self._extract_section(text, ["EVALUATION CRITERIA", "EVALUATION"])

            return ProcessedTender(
                tender_id=tender_id,
                reference_number=raw.get("reference_number", ""),
                organisation=raw.get("organisation", ""),
                title=raw.get("title", ""),
                category=raw.get("category", ""),
                tender_value_raw=raw.get("tender_value", ""),
                tender_value_numeric=self._parse_inr(raw.get("tender_value", "")),
                emd_amount_raw=raw.get("emd_amount", ""),
                emd_amount_numeric=self._parse_inr(raw.get("emd_amount", "")),
                submission_deadline=raw.get("submission_deadline", ""),
                scope=scope,
                eligibility=eligibility,
                evaluation_criteria=eval_section,
                extracted_entities=entities,
                source_url=raw.get("source_url", ""),
                processing_success=True,
            )

        except Exception as e:
            logger.warning(f"  [{tender_id}] Error: {e}")
            return ProcessedTender(
                tender_id=tender_id,
                reference_number=raw.get("reference_number", ""),
                organisation=raw.get("organisation", ""),
                title=raw.get("title", ""),
                category=raw.get("category", ""),
                tender_value_raw=raw.get("tender_value", ""),
                tender_value_numeric=None,
                emd_amount_raw=raw.get("emd_amount", ""),
                emd_amount_numeric=None,
                submission_deadline=raw.get("submission_deadline", ""),
                scope="",
                eligibility="",
                evaluation_criteria="",
                extracted_entities=[],
                source_url=raw.get("source_url", ""),
                processing_success=False,
                error_message=str(e),
            )

    def _get_text(self, raw):
        """
        Returns the best available text for a tender.

        Priority order:
        1. raw_text stored in the JSON record (scraped from the portal page
           by the Download Agent — present for all records).
        2. PDF text extracted from the first PDF in the tender's download folder.
        3. Minimal fallback: concatenation of title + category + organisation.
        """
        if raw.get("raw_text"):
            return raw["raw_text"]

        download_path = raw.get("download_path", "")
        if download_path:
            folder = Path(download_path)
            if folder.exists():
                pdf_text = self._extract_pdf_text(folder)
                if pdf_text:
                    return pdf_text

        return f"{raw.get('title', '')} {raw.get('category', '')} {raw.get('organisation', '')}"

    def _extract_pdf_text(self, folder):
        """
        Uses pdfplumber to extract text from the first PDF found in a folder.
        Concatenates all pages and returns up to ~2,700 words.
        """
        for pdf_path in sorted(folder.glob("*.pdf")):
            try:
                pages = []
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            pages.append(page_text)
                full_text = "\n".join(pages)
                logger.info(f"    Extracted {len(full_text)} chars from {pdf_path.name}")
                return full_text[:MAX_CHUNK_WORDS * 6]
            except Exception as e:
                logger.warning(f"    PDF extraction failed for {pdf_path.name}: {e}")

        return ""

    def _run_ner(self, text):
        """
        Runs the HuggingFace NER pipeline over the text, chunked into
        MAX_CHUNK_WORDS-word blocks to respect BERT's 512-token limit.

        Returns:
            List of entity dicts: {entity_group, score, word, start, end}
        """
        words = text.split()
        chunks = []
        for i in range(0, len(words), MAX_CHUNK_WORDS):
            chunks.append(" ".join(words[i:i + MAX_CHUNK_WORDS]))

        all_entities = []
        for chunk in chunks:
            try:
                entities = self.ner_pipeline(chunk)
                for ent in entities:
                    ent["score"] = float(ent["score"])
                all_entities.extend(entities)
            except Exception as e:
                logger.warning(f"    NER chunk failed: {e}")

        return all_entities

    def _extract_section(self, text, headers):
        """
        Finds a labelled section in the text by matching against the given header
        strings, then returns the following paragraph (up to 600 characters).

        Args:
            text:    Full document text.
            headers: List of possible header strings (tried in order).

        Returns:
            Extracted section text, or empty string if not found.
        """
        for header in headers:
            pattern = re.compile(
                r'(?i)' + re.escape(header) + r'[:\s]*\n(.*?)(?=\n[A-Z][A-Z\s]{5,}:|\Z)',
                re.DOTALL,
            )
            match = pattern.search(text)
            if match:
                return match.group(1).strip()[:600]
        return ""

    def _parse_inr(self, value_str):
        """
        Parses an INR amount string (e.g. "2,40,00,000") into a float.
        Returns None if parsing fails.
        """
        if not value_str:
            return None
        try:
            cleaned = re.sub(r"[^\d.]", "", str(value_str))
            return float(cleaned) if cleaned else None
        except ValueError:
            return None

    def _save_output(self):
        output_path = self.data_dir / "processed_tenders.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: {output_path}")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Document Processing Agent")
    parser.add_argument("--data_dir", default="./tender_data",
                        help="Directory containing tender_index.json")
    parser.add_argument("--model", default=MODEL_PATH,
                        help="HuggingFace model ID or path to fine-tuned model")
    args = parser.parse_args()

    agent = DocumentProcessingAgent(data_dir=args.data_dir, model_path=args.model)
    results = agent.run()

    successful = [r for r in results if r.processing_success]
    print(f"\nSummary: {len(successful)}/{len(results)} tenders processed successfully")
    print(f"Output:  {args.data_dir}/processed_tenders.json")
