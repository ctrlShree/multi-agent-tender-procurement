# Multi-Agent System for Intelligent Tender Procurement

**CS 5100 — Foundations of Artificial Intelligence**  
Shubham Bagdare (002309528) · Shreyash Sawant (002598621)

---

## Overview

A multi-agent AI system that automates the discovery, analysis, and recommendation
of government procurement tenders for businesses. Given a company profile describing
capabilities and industry focus, the system:

1. **Scrapes** tender documents from eTenders.gov.in using Selenium
2. **Extracts** structured information from PDFs using a fine-tuned BERT NER model
3. **Ranks** tenders by semantic relevance, budget fit, and deadline feasibility using FAISS
4. **Answers** natural language queries via a RAG pipeline (FAISS + Llama 3.1 via Groq)

---

## Repository Structure

```
tender_system/
├── orchestrator.py                # Coordinates all 4 agents end-to-end
├── download_agent.py              # Agent 1 — Selenium scraper for eTenders.gov.in
├── scraper.py                     # Standalone scraper (prototype; kept for reference)
├── document_processing_agent.py   # Agent 2 — PDF extraction + BERT NER
├── analysis_agent.py              # Agent 3 — sentence-transformers + FAISS scoring
├── conversational_agent.py        # Agent 4 — RAG + Groq Llama 3.1
├── fine_tune_ner.py               # NER fine-tuning script (run after annotation)
└── requirements.txt
```

Data directory produced after running (default: `./tender_data/`):
```
tender_data/
├── tender_index.json          # Download Agent output — one record per tender
├── processed_tenders.json     # Processing Agent output — NER-enriched records
├── ranked_tenders.json        # Analysis Agent output — scored and ranked
├── faiss_index.bin            # FAISS vector index (used by Conversational Agent)
├── faiss_records.pkl          # Ordered tender list matching FAISS row indices
└── pipeline_status.json       # Stage-by-stage timing and status log
```

---

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY="your_key_here"   # Free key at https://console.groq.com
```

---

## Running the Pipeline

### Full run (from scratch)

```bash
python orchestrator.py \
    --data_dir ./tender_data \
    --profile "cloud computing, AWS infrastructure, cybersecurity, IT consulting"
```

This launches Chrome, navigates eTenders.gov.in, and prompts for CAPTCHA resolution
once. All subsequent navigation is automated. After downloading, it runs NER extraction,
FAISS indexing, and opens an interactive conversational CLI.

### Skipping the download stage (data already collected)

```bash
python orchestrator.py --skip_download --data_dir ./tender_data
```

### Skipping both download and processing (using existing processed_tenders.json)

```bash
python orchestrator.py --skip_download --skip_processing --data_dir ./tender_data
```

---

## Running Agents Individually

```bash
# Agent 1 — Download
python download_agent.py

# Agent 2 — Document Processing
python document_processing_agent.py --data_dir ./tender_data
# Use fine-tuned model once available:
python document_processing_agent.py --data_dir ./tender_data --model ./ner_model_finetuned

# Agent 3 — Analysis and Recommendation
python analysis_agent.py --data_dir ./tender_data \
    --profile "cloud computing, AWS, cybersecurity"

# Agent 4 — Conversational Interface (interactive CLI)
python conversational_agent.py --data_dir ./tender_data
```

---

## Fine-Tuning the NER Model

Once Label Studio annotation is complete, export to JSONL (one document per line):

```json
{"tokens": ["Tender", "Reference", ":", "MCGM/IT/2024/1143"], "ner_tags": ["O", "O", "O", "B-TENDER_REF"]}
```

Then run:

```bash
python fine_tune_ner.py \
    --train_file ./annotations/train.jsonl \
    --eval_file  ./annotations/eval.jsonl \
    --output_dir ./ner_model_finetuned \
    --epochs 5
```

The fine-tuned model replaces `dslim/bert-base-NER` in the processing agent:

```bash
python document_processing_agent.py --model ./ner_model_finetuned --data_dir ./tender_data
```

---

## Agent Details

| Agent | Input | Output | Key Technology |
|-------|-------|--------|----------------|
| Download Agent | Company profile keywords | `tender_index.json` + per-tender folders | Selenium, eTenders.gov.in |
| Document Processing Agent | `tender_index.json` + PDFs | `processed_tenders.json` | `dslim/bert-base-NER` (fine-tuned), pdfplumber |
| Analysis Agent | `processed_tenders.json` | `ranked_tenders.json`, FAISS index | `all-MiniLM-L6-v2`, FAISS |
| Conversational Agent | FAISS index + user query | Natural language response | RAG, Groq Llama 3.1 |

### Composite scoring formula (Analysis Agent)

```
score = 0.40 × semantic_similarity   (cosine, company profile vs tender scope)
      + 0.25 × budget_fit            (normalized match to target contract range)
      + 0.20 × deadline_score        (penalizes imminent/passed deadlines)
      + 0.15 × category_overlap      (keyword match, profile vs tender category)
```

---

## Data Source

**Current:** eTenders.gov.in — India's Central Public Procurement Portal.
80+ government organisations; documents freely downloadable as ZIP files (PDFs + BoQ sheets).

**Planned:** SAM.gov — US federal procurement portal.
System Account API access pending institutional approval.
The NER models and pipeline transfer to SAM.gov data with minimal modification.

---

## Key References

- Lima et al. (2025): BERT-based NER for procurement text — establishes F1 = 90.5% benchmark at our annotation scale
- Reimers & Gurevych (2019): sentence-transformers (all-MiniLM-L6-v2)
- Lewis et al. (2020): Retrieval-Augmented Generation (RAG)
- dslim/bert-base-NER: https://huggingface.co/dslim/bert-base-NER
- all-MiniLM-L6-v2: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
