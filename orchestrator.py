"""
Orchestrator
=============
Coordinates the full multi-agent pipeline end-to-end:

    DownloadAgent  →  DocumentProcessingAgent  →  AnalysisAgent  →  ConversationalAgent

Each agent reads the output of the previous stage and writes its own to
the shared data directory. The orchestrator validates prerequisites before
each stage, records timing and status, and handles failures gracefully.

Usage:
    # Full run (requires Chrome + manual CAPTCHA on first download)
    python orchestrator.py --data_dir ./tender_data

    # Skip download if tender_index.json already exists
    python orchestrator.py --data_dir ./tender_data --skip_download

    # Skip both download and processing (use existing processed_tenders.json)
    python orchestrator.py --data_dir ./tender_data --skip_download --skip_processing
"""

import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    stage: str
    status: str           # "pending" | "running" | "done" | "skipped" | "failed"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    summary: str = ""
    error: str = ""


@dataclass
class PipelineStatus:
    profile: str
    data_dir: str
    stages: list = field(default_factory=lambda: [
        StageResult("download",       "pending"),
        StageResult("processing",     "pending"),
        StageResult("analysis",       "pending"),
        StageResult("conversational", "pending"),
    ])

    def get(self, name):
        for s in self.stages:
            if s.stage == name:
                return s
        return None

    def to_dict(self):
        return {
            "profile":  self.profile,
            "data_dir": self.data_dir,
            "stages": [
                {k: getattr(s, k) for k in
                 ["stage", "status", "started_at", "finished_at", "summary", "error"]}
                for s in self.stages
            ],
        }


class Orchestrator:
    """
    Coordinates the four specialized agents in sequence, validating
    prerequisites and recording stage outcomes.

    After run() completes, the conversational agent is kept alive
    so callers can invoke orchestrator.query() to chat with the corpus.
    """

    def __init__(self, data_dir, company_profile,
                 groq_api_key=None,
                 budget_min=50_00_000,
                 budget_max=5_00_00_000,
                 max_orgs=30,
                 max_tenders=150):
        """
        Args:
            data_dir:        Base directory for all agent inputs/outputs.
            company_profile: Capability description used for filtering and scoring.
            groq_api_key:    Groq API key for the conversational agent.
            budget_min:      Lower bound of the company's target contract value (INR).
            budget_max:      Upper bound of the company's target contract value (INR).
            max_orgs:        Max organisations the download agent visits.
            max_tenders:     Total tender download cap.
        """
        self.data_dir       = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.company_profile = company_profile
        self.groq_api_key   = groq_api_key or os.environ.get("GROQ_API_KEY")
        self.budget_min     = budget_min
        self.budget_max     = budget_max
        self.max_orgs       = max_orgs
        self.max_tenders    = max_tenders
        self.status         = PipelineStatus(profile=company_profile, data_dir=str(data_dir))
        self.conversational_agent = None

    def run(self, skip_download=False, skip_processing=False):
        """
        Runs the full pipeline. Pass skip_download=True or skip_processing=True
        to reuse data from a previous run.

        Returns:
            PipelineStatus summarising each stage's result.
        """
        logger.info("=" * 60)
        logger.info(f"Orchestrator starting — profile: {self.company_profile}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info("=" * 60)

        if skip_download:
            self._mark_skipped("download", "skipped by --skip_download flag")
        else:
            self._run_download()

        if skip_processing:
            self._mark_skipped("processing", "skipped by --skip_processing flag")
        elif not self._exists("tender_index.json"):
            self._mark_failed("processing",
                              "tender_index.json not found — run download stage first")
        else:
            self._run_processing()

        if not self._exists("processed_tenders.json"):
            self._mark_failed("analysis",
                              "processed_tenders.json not found — run processing stage first")
        else:
            self._run_analysis()

        if not self._exists("faiss_index.bin"):
            self._mark_failed("conversational",
                              "faiss_index.bin not found — run analysis stage first")
        else:
            self._run_conversational()

        self._save_status()
        self._print_summary()
        return self.status

    def query(self, user_message):
        """
        Routes a user query to the conversational agent.
        Must be called after run() has completed successfully.

        Returns:
            The assistant's response string.
        """
        if self.conversational_agent is None:
            raise RuntimeError(
                "Conversational agent not initialized. "
                "Ensure the pipeline completed successfully."
            )
        return self.conversational_agent.query(user_message)

    # -------------------------------------------------------------------------

    def _run_download(self):
        stage = self.status.get("download")
        stage.status = "running"
        stage.started_at = _now()
        try:
            from download_agent import DownloadAgent

            agent = DownloadAgent(
                download_dir=str(self.data_dir),
                company_profile=self.company_profile,
                max_orgs=self.max_orgs,
                max_tenders_per_org=5,
            )
            agent.start_session()
            results = agent.run(max_tenders=self.max_tenders)
            agent.close()

            ok = sum(1 for r in results if r.download_success)
            stage.status      = "done"
            stage.finished_at = _now()
            stage.summary     = f"{ok}/{len(results)} tenders downloaded"
        except Exception as e:
            self._mark_failed("download", str(e))

    def _run_processing(self):
        stage = self.status.get("processing")
        stage.status = "running"
        stage.started_at = _now()
        try:
            from document_processing_agent import DocumentProcessingAgent

            agent = DocumentProcessingAgent(data_dir=str(self.data_dir))
            results = agent.run()

            ok = sum(1 for r in results if r.processing_success)
            stage.status      = "done"
            stage.finished_at = _now()
            stage.summary     = f"{ok}/{len(results)} tenders processed"
        except Exception as e:
            self._mark_failed("processing", str(e))

    def _run_analysis(self):
        stage = self.status.get("analysis")
        stage.status = "running"
        stage.started_at = _now()
        try:
            from analysis_agent import AnalysisAgent

            agent = AnalysisAgent(
                data_dir=str(self.data_dir),
                company_profile=self.company_profile,
                budget_min=self.budget_min,
                budget_max=self.budget_max,
            )
            results = agent.run()

            top = results[0] if results else None
            stage.status      = "done"
            stage.finished_at = _now()
            stage.summary     = (
                f"{len(results)} tenders ranked. "
                f"Top: '{top.title[:50]}' (score={top.composite_score:.2f})"
                if top else "0 tenders ranked"
            )
        except Exception as e:
            self._mark_failed("analysis", str(e))

    def _run_conversational(self):
        stage = self.status.get("conversational")
        stage.status = "running"
        stage.started_at = _now()
        try:
            if not self.groq_api_key:
                raise ValueError(
                    "GROQ_API_KEY not set. Export the variable or pass --groq_api_key."
                )
            from conversational_agent import ConversationalAgent

            agent = ConversationalAgent(
                data_dir=str(self.data_dir),
                groq_api_key=self.groq_api_key,
            )
            agent.load_index()
            self.conversational_agent = agent

            stage.status      = "done"
            stage.finished_at = _now()
            stage.summary     = f"RAG pipeline ready — {len(agent.tender_records)} tenders indexed"
        except Exception as e:
            self._mark_failed("conversational", str(e))

    # -------------------------------------------------------------------------

    def _exists(self, filename):
        return (self.data_dir / filename).exists()

    def _mark_skipped(self, name, reason):
        s = self.status.get(name)
        s.status  = "skipped"
        s.summary = reason

    def _mark_failed(self, name, error):
        s = self.status.get(name)
        s.status      = "failed"
        s.finished_at = _now()
        s.error       = error
        logger.error(f"Stage '{name}' failed: {error}")

    def _save_status(self):
        path = self.data_dir / "pipeline_status.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.status.to_dict(), f, indent=2)
        logger.info(f"Pipeline status saved: {path}")

    def _print_summary(self):
        icons = {"done": "✓", "skipped": "–", "failed": "✗", "pending": "?", "running": "…"}
        print("\n" + "=" * 60)
        print("  Pipeline Summary")
        print("=" * 60)
        for s in self.status.stages:
            icon = icons.get(s.status, "?")
            detail = s.summary or s.error
            print(f"  [{icon}] {s.stage:<15}  {s.status:<10}  {detail}")
        print("=" * 60)


def _now():
    return datetime.now().isoformat(timespec="seconds")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full tender procurement pipeline")
    parser.add_argument("--data_dir",       default="./tender_data")
    parser.add_argument("--profile",
                        default="cloud computing, AWS infrastructure, cybersecurity, IT consulting",
                        help="Company capability profile (comma-separated)")
    parser.add_argument("--skip_download",  action="store_true",
                        help="Skip the Download Agent (use existing tender_index.json)")
    parser.add_argument("--skip_processing", action="store_true",
                        help="Skip NER processing (use existing processed_tenders.json)")
    parser.add_argument("--groq_api_key",   default=None)
    parser.add_argument("--budget_min",     type=int, default=50_00_000)
    parser.add_argument("--budget_max",     type=int, default=5_00_00_000)
    parser.add_argument("--max_orgs",       type=int, default=30)
    parser.add_argument("--max_tenders",    type=int, default=150)
    args = parser.parse_args()

    orch = Orchestrator(
        data_dir=args.data_dir,
        company_profile=args.profile,
        groq_api_key=args.groq_api_key,
        budget_min=args.budget_min,
        budget_max=args.budget_max,
        max_orgs=args.max_orgs,
        max_tenders=args.max_tenders,
    )
    orch.run(
        skip_download=args.skip_download,
        skip_processing=args.skip_processing,
    )

    # If the conversational agent initialized, drop into interactive CLI
    if orch.conversational_agent is not None:
        print("\nPipeline complete. Dropping into conversational interface...\n")
        from conversational_agent import run_interactive
        run_interactive(data_dir=args.data_dir, groq_api_key=args.groq_api_key)
