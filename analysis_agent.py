"""
Analysis and Recommendation Agent
===================================
Reads processed_tenders.json, embeds each tender using sentence-transformers,
builds a FAISS index, computes composite relevance scores against a company
profile, and writes ranked_tenders.json.

The FAISS index is persisted to disk so the Conversational Agent can load it
directly for retrieval-augmented generation.

Composite score (weighted sum):
    semantic_score  × 0.40  (cosine similarity, company profile vs tender scope)
    budget_score    × 0.25  (normalized fit against target contract value range)
    deadline_score  × 0.20  (penalty for imminent or passed deadlines)
    category_score  × 0.15  (keyword overlap, profile vs tender category + title)

Run:
    python analysis_agent.py --data_dir ./tender_data
    python analysis_agent.py --data_dir ./tender_data --profile "cloud, AWS, cybersecurity"
"""

import json
import pickle
import logging
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BUDGET_MIN = 50_00_000      # INR 50 lakh
DEFAULT_BUDGET_MAX = 5_00_00_000    # INR 5 crore


@dataclass
class RankedTender:
    """A processed tender enriched with relevance scores and final rank."""
    rank: int
    tender_id: str
    reference_number: str
    organisation: str
    title: str
    category: str
    tender_value_raw: str
    tender_value_numeric: Optional[float]
    emd_amount_raw: str
    submission_deadline: str
    scope: str
    eligibility: str
    source_url: str
    semantic_score: float
    budget_score: float
    deadline_score: float
    category_score: float
    composite_score: float


class AnalysisAgent:
    """
    Agent responsible for:
    1. Embedding all tender scopes using sentence-transformers
    2. Building and persisting a FAISS index for downstream RAG retrieval
    3. Computing composite relevance scores per tender against the company profile
    4. Ranking tenders and writing ranked_tenders.json
    """

    def __init__(self, data_dir, company_profile,
                 budget_min=DEFAULT_BUDGET_MIN,
                 budget_max=DEFAULT_BUDGET_MAX,
                 model_name=EMBEDDING_MODEL):
        """
        Args:
            data_dir:        Directory containing processed_tenders.json.
            company_profile: Plain-text company capability description,
                             e.g. "cloud computing, AWS, cybersecurity, IT consulting".
            budget_min:      Lower bound of the company's target contract size (INR).
            budget_max:      Upper bound of the company's target contract size (INR).
            model_name:      SentenceTransformer model for embeddings.
        """
        self.data_dir = Path(data_dir)
        self.company_profile = company_profile
        self.budget_min = budget_min
        self.budget_max = budget_max
        self.model_name = model_name
        self.model = None
        self.faiss_index = None
        self.tender_records = []    # ordered list — row i in FAISS = tender_records[i]
        self.results = []           # list of RankedTender

    def load_model(self):
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info("Embedding model loaded.")

    def run(self):
        """
        Scores and ranks all successfully processed tenders.

        Returns:
            List of RankedTender objects sorted by composite_score descending.
        """
        if self.model is None:
            self.load_model()

        input_path = self.data_dir / "processed_tenders.json"
        if not input_path.exists():
            raise FileNotFoundError(
                f"processed_tenders.json not found in {self.data_dir}. "
                "Run the Document Processing Agent first."
            )

        with open(input_path, "r", encoding="utf-8") as f:
            all_tenders = json.load(f)

        tenders = [t for t in all_tenders if t.get("processing_success")]
        logger.info(f"Scoring {len(tenders)} successfully processed tenders...")

        self._build_faiss_index(tenders)

        profile_vec = self.model.encode([self.company_profile], convert_to_numpy=True)[0]
        profile_keywords = [kw.strip().lower() for kw in self.company_profile.split(",")]

        scored = []
        for i, tender in enumerate(self.tender_records):
            tender_vec = self._get_vector(i)
            semantic  = float(self._cosine(profile_vec, tender_vec))
            budget    = self._budget_score(tender.get("tender_value_numeric"))
            deadline  = self._deadline_score(tender.get("submission_deadline", ""))
            category  = self._category_score(
                tender.get("category", "") + " " + tender.get("title", ""),
                profile_keywords,
            )
            composite = (
                0.40 * semantic +
                0.25 * budget   +
                0.20 * deadline +
                0.15 * category
            )
            scored.append((composite, semantic, budget, deadline, category, tender))

        scored.sort(key=lambda x: x[0], reverse=True)

        self.results = []
        for rank, (composite, semantic, budget, deadline, category, t) in enumerate(scored, start=1):
            self.results.append(RankedTender(
                rank=rank,
                tender_id=t.get("tender_id", ""),
                reference_number=t.get("reference_number", ""),
                organisation=t.get("organisation", ""),
                title=t.get("title", ""),
                category=t.get("category", ""),
                tender_value_raw=t.get("tender_value_raw", ""),
                tender_value_numeric=t.get("tender_value_numeric"),
                emd_amount_raw=t.get("emd_amount_raw", ""),
                submission_deadline=t.get("submission_deadline", ""),
                scope=t.get("scope", ""),
                eligibility=t.get("eligibility", ""),
                source_url=t.get("source_url", ""),
                semantic_score=round(semantic, 4),
                budget_score=round(budget, 4),
                deadline_score=round(deadline, 4),
                category_score=round(category, 4),
                composite_score=round(composite, 4),
            ))

        self._save_output()
        self._save_faiss_index()
        logger.info(f"Done. Top tender: '{self.results[0].title}'" if self.results else "No results.")
        return self.results

    def retrieve_top_k(self, query, k=5):
        """
        Semantic nearest-neighbour retrieval over the FAISS index.
        Used by the Conversational Agent for RAG retrieval.

        Args:
            query: Natural language query string.
            k:     Number of top results to return.

        Returns:
            List of tender dicts from tender_records, ordered by similarity.
        """
        if self.faiss_index is None:
            raise RuntimeError("FAISS index not built. Call run() first.")
        query_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        k = min(k, len(self.tender_records))
        _, indices = self.faiss_index.search(query_vec, k)
        return [self.tender_records[i] for i in indices[0] if i < len(self.tender_records)]

    # -------------------------------------------------------------------------

    def _build_faiss_index(self, tenders):
        """
        Embeds all tender texts (title + scope + category) and builds a flat
        L2 FAISS index. Stores the tender list in self.tender_records so that
        FAISS row i always maps to tender_records[i].
        """
        self.tender_records = tenders
        texts = [
            f"{t.get('title', '')}. {t.get('scope', '')}. {t.get('category', '')}"
            for t in tenders
        ]
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dim)
        self.faiss_index.add(embeddings.astype("float32"))
        logger.info(f"FAISS index built: {len(tenders)} vectors, dim={dim}")

    def _get_vector(self, index):
        """Retrieves the stored embedding for a tender by its FAISS index position."""
        vec = np.zeros(self.faiss_index.d, dtype="float32")
        self.faiss_index.reconstruct(index, vec)
        return vec

    def _cosine(self, a, b):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    def _budget_score(self, value_numeric):
        """
        Returns 1.0 if the tender value falls within [budget_min, budget_max].
        Decays linearly outside that range, floored at 0.1.
        Returns 0.5 (neutral) when value is missing.
        """
        if value_numeric is None:
            return 0.5
        if self.budget_min <= value_numeric <= self.budget_max:
            return 1.0
        mid = (self.budget_min + self.budget_max) / 2
        distance_ratio = abs(value_numeric - mid) / mid
        return max(0.1, round(1.0 - distance_ratio * 0.5, 4))

    def _deadline_score(self, deadline_str):
        """
        Penalizes tenders with imminent or passed deadlines.

        Score ladder:
            > 30 days  →  1.0
            15–30 days →  0.8
            7–15 days  →  0.6
            2–7 days   →  0.4
            < 2 days   →  0.1
            Passed     →  0.0
            Unknown    →  0.5 (neutral)
        """
        if not deadline_str:
            return 0.5
        formats = ["%Y-%m-%d", "%d-%b-%Y", "%d/%m/%Y", "%d-%m-%Y"]
        deadline_date = None
        for fmt in formats:
            try:
                deadline_date = datetime.strptime(deadline_str, fmt).date()
                break
            except ValueError:
                continue
        if deadline_date is None:
            return 0.5
        days = (deadline_date - date.today()).days
        if days < 0:   return 0.0
        if days < 2:   return 0.1
        if days < 7:   return 0.4
        if days < 15:  return 0.6
        if days < 30:  return 0.8
        return 1.0

    def _category_score(self, text, profile_keywords):
        """Fraction of company profile keywords present in the text. Returns 0.0–1.0."""
        if not profile_keywords:
            return 0.0
        text_lower = text.lower()
        matches = sum(1 for kw in profile_keywords if kw in text_lower)
        return round(matches / len(profile_keywords), 4)

    def _save_output(self):
        output_path = self.data_dir / "ranked_tenders.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: {output_path}")

    def _save_faiss_index(self):
        """
        Saves the FAISS index binary and the ordered tender records list so the
        Conversational Agent can load them without re-running the analysis step.
        """
        faiss.write_index(self.faiss_index, str(self.data_dir / "faiss_index.bin"))
        with open(self.data_dir / "faiss_records.pkl", "wb") as f:
            pickle.dump(self.tender_records, f)
        logger.info(f"FAISS index saved to {self.data_dir}/faiss_index.bin")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Analysis and Recommendation Agent")
    parser.add_argument("--data_dir", default="./tender_data")
    parser.add_argument("--profile",
                        default="cloud computing, AWS infrastructure, cybersecurity, IT consulting",
                        help="Company capability profile (comma-separated)")
    parser.add_argument("--budget_min", type=int, default=DEFAULT_BUDGET_MIN)
    parser.add_argument("--budget_max", type=int, default=DEFAULT_BUDGET_MAX)
    args = parser.parse_args()

    agent = AnalysisAgent(
        data_dir=args.data_dir,
        company_profile=args.profile,
        budget_min=args.budget_min,
        budget_max=args.budget_max,
    )
    results = agent.run()

    print(f"\nTop 5 Recommendations:")
    for r in results[:5]:
        print(f"  #{r.rank:2d}  [{r.composite_score:.2f}]  {r.title[:60]}")
        print(f"       {r.organisation}")
        print(f"       Value: {r.tender_value_raw}  |  Deadline: {r.submission_deadline}")
