"""
Conversational Interface Agent
================================
Implements a Retrieval-Augmented Generation (RAG) pipeline over the
processed tender corpus. On each user query:
    1. Embed the query using the same sentence-transformer model.
    2. Retrieve the top-K most similar tenders from the FAISS index.
    3. Format retrieved tender context into a prompt.
    4. Call Llama 3.1 (8B) via the Groq API to generate a grounded response.
    5. Maintain conversation history for multi-turn interactions.

Setup:
    Set the GROQ_API_KEY environment variable before running:
        export GROQ_API_KEY="your_groq_api_key_here"
    Free-tier API keys are available at https://console.groq.com

Run:
    python conversational_agent.py --data_dir ./tender_data
"""

import os
import json
import pickle
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL      = "llama-3.1-8b-instant"   # free tier on Groq
TOP_K_RETRIEVE  = 5                         # number of tenders retrieved per query
MAX_HISTORY     = 10                        # max conversation turns kept in context

SYSTEM_PROMPT = (
    "You are an intelligent procurement assistant for a technology company. "
    "You have access to a curated database of government tender opportunities. "
    "When answering questions:\n"
    "- Always ground your answer in the retrieved tender information provided.\n"
    "- Be specific: reference tender IDs, organisations, values, and deadlines.\n"
    "- If comparing tenders, use a structured format.\n"
    "- If a tender does not match the query well, say so clearly.\n"
    "- Keep responses concise but complete.\n"
    "- Do not invent tender details not present in the context."
)


class ConversationalAgent:
    """
    Agent responsible for:
    1. Loading the FAISS index and tender records built by the Analysis Agent.
    2. Answering natural language queries about tenders via RAG + Llama 3.1.
    3. Supporting multi-turn conversation with conversation history.
    """

    def __init__(self, data_dir, groq_api_key=None, top_k=TOP_K_RETRIEVE):
        """
        Args:
            data_dir:     Directory containing faiss_index.bin, faiss_records.pkl,
                          and ranked_tenders.json.
            groq_api_key: Groq API key. Falls back to GROQ_API_KEY env variable.
            top_k:        Number of tenders to retrieve per query.
        """
        self.data_dir = Path(data_dir)
        self.top_k = top_k
        self.conversation_history = []

        api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "No Groq API key found. Set the GROQ_API_KEY environment variable "
                "or pass groq_api_key= to ConversationalAgent().\n"
                "Get a free key at https://console.groq.com"
            )
        self.groq_client = Groq(api_key=api_key)

        self.embedding_model = None
        self.faiss_index = None
        self.tender_records = []
        self.ranked_tenders = []

    def load_index(self):
        """
        Loads the FAISS index, tender records, and ranked results from disk.
        Also loads the embedding model.
        """
        index_path   = self.data_dir / "faiss_index.bin"
        records_path = self.data_dir / "faiss_records.pkl"
        ranked_path  = self.data_dir / "ranked_tenders.json"

        if not index_path.exists():
            raise FileNotFoundError(
                f"faiss_index.bin not found in {self.data_dir}. "
                "Run the Analysis Agent first."
            )

        logger.info("Loading FAISS index and tender records...")
        self.faiss_index = faiss.read_index(str(index_path))

        with open(records_path, "rb") as f:
            self.tender_records = pickle.load(f)

        if ranked_path.exists():
            with open(ranked_path, "r", encoding="utf-8") as f:
                self.ranked_tenders = json.load(f)

        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"Ready — {len(self.tender_records)} tenders indexed.")

    def query(self, user_message):
        """
        Handles a single user query end-to-end:
            1. Retrieve top-K relevant tenders via FAISS.
            2. Build a prompt with retrieved context + conversation history.
            3. Call Llama 3.1 via Groq.
            4. Append the exchange to conversation history.

        Args:
            user_message: The user's natural language question.

        Returns:
            The assistant's response string.
        """
        if self.faiss_index is None:
            self.load_index()

        retrieved = self._retrieve(user_message)
        context   = self._format_context(retrieved)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.conversation_history[-(MAX_HISTORY * 2):])
        messages.append({
            "role": "user",
            "content": f"Retrieved tender context:\n{context}\n\nUser question: {user_message}",
        })

        response_text = self._call_groq(messages)

        self.conversation_history.append({"role": "user",      "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": response_text})

        return response_text

    def reset_history(self):
        """Clears the conversation history to start a fresh session."""
        self.conversation_history = []
        logger.info("Conversation history cleared.")

    def get_top_recommendations(self, n=5):
        """
        Returns the top-N ranked tenders as a formatted string.
        Useful as an opening context message in the chat interface.
        """
        if not self.ranked_tenders:
            return "No ranked tenders available. Run the Analysis Agent first."
        lines = [f"Top {n} recommended tenders:\n"]
        for t in self.ranked_tenders[:n]:
            lines.append(
                f"#{t['rank']}. {t['title']}\n"
                f"   Org: {t['organisation']}\n"
                f"   Value: {t['tender_value_raw']}  |  Deadline: {t['submission_deadline']}\n"
                f"   Relevance: {int(t['composite_score'] * 100)}%\n"
            )
        return "\n".join(lines)

    # -------------------------------------------------------------------------

    def _retrieve(self, query):
        """
        Embeds the query and searches the FAISS index for the top-K most
        similar tender documents.
        """
        query_vec = self.embedding_model.encode([query], convert_to_numpy=True).astype("float32")
        k = min(self.top_k, len(self.tender_records))
        _, indices = self.faiss_index.search(query_vec, k)
        return [self.tender_records[i] for i in indices[0] if i < len(self.tender_records)]

    def _format_context(self, tenders):
        """Formats retrieved tender dicts into a plain-text prompt context block."""
        parts = []
        for i, t in enumerate(tenders, start=1):
            parts.append(
                f"[Tender {i}]\n"
                f"ID: {t.get('tender_id', 'N/A')}  |  Ref: {t.get('reference_number', 'N/A')}\n"
                f"Title: {t.get('title', 'N/A')}\n"
                f"Organisation: {t.get('organisation', 'N/A')}\n"
                f"Category: {t.get('category', 'N/A')}\n"
                f"Value: INR {t.get('tender_value_raw', 'N/A')}\n"
                f"EMD: INR {t.get('emd_amount_raw', 'N/A')}\n"
                f"Deadline: {t.get('submission_deadline', 'N/A')}\n"
                f"Scope: {t.get('scope', 'N/A')[:400]}\n"
                f"Eligibility: {t.get('eligibility', 'N/A')[:300]}\n"
            )
        return "\n---\n".join(parts)

    def _call_groq(self, messages):
        """Calls Llama 3.1 via Groq and returns the response text."""
        try:
            response = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            return f"Error: could not get a response from the language model. Details: {e}"


# ── Interactive CLI ────────────────────────────────────────────────────────────

def run_interactive(data_dir, groq_api_key=None):
    """Runs a simple interactive command-line chat session."""
    agent = ConversationalAgent(data_dir=data_dir, groq_api_key=groq_api_key)
    agent.load_index()

    print("\n" + "=" * 60)
    print("  Tender Procurement — Conversational Interface")
    print("=" * 60)
    print(agent.get_top_recommendations())
    print("Type your question. Commands: 'reset' to clear history | 'quit' to exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye.")
            break
        if user_input.lower() == "reset":
            agent.reset_history()
            print("Conversation reset.\n")
            continue

        response = agent.query(user_input)
        print(f"\nAssistant: {response}\n")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Conversational Interface Agent")
    parser.add_argument("--data_dir", default="./tender_data")
    parser.add_argument("--groq_api_key", default=None,
                        help="Groq API key (or set GROQ_API_KEY env variable)")
    args = parser.parse_args()

    run_interactive(data_dir=args.data_dir, groq_api_key=args.groq_api_key)
