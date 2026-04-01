"""
NER Fine-Tuning Script
========================
Fine-tunes dslim/bert-base-NER on manually annotated procurement documents
exported from Label Studio.

Entity types in our annotation schema (BIO encoding):
    TENDER_REF     — tender reference number (e.g. MCGM/IT/2024/1143)
    ORG_NAME       — procuring organisation name
    BUDGET         — estimated contract value / tender value
    EMD            — earnest money deposit amount
    DEADLINE       — bid submission or opening date
    SCOPE          — project scope / work description
    ELIGIBILITY    — eligibility requirement spans
    EVAL_CRITERIA  — evaluation criteria spans

Expected input format — JSON Lines, one document per line:
    {"tokens": ["Tender", "Ref", ":", "MCGM/IT/2024/1143"], "ner_tags": ["O", "O", "O", "B-TENDER_REF"]}

Label Studio export:
    Export → CoNLL 2003 format, then convert to JSONL with the helper below,
    OR export directly as JSONL if using a custom template.

Run:
    python fine_tune_ner.py \
        --train_file ./annotations/train.jsonl \
        --eval_file  ./annotations/eval.jsonl \
        --output_dir ./ner_model_finetuned \
        --epochs 5

After training, update document_processing_agent.py:
    agent = DocumentProcessingAgent(data_dir=..., model_path="./ner_model_finetuned")
"""

import os
import json
import logging
import argparse
import numpy as np
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Label schema ──────────────────────────────────────────────────────────────

LABEL_LIST = [
    "O",
    "B-TENDER_REF",   "I-TENDER_REF",
    "B-ORG_NAME",     "I-ORG_NAME",
    "B-BUDGET",       "I-BUDGET",
    "B-EMD",          "I-EMD",
    "B-DEADLINE",     "I-DEADLINE",
    "B-SCOPE",        "I-SCOPE",
    "B-ELIGIBILITY",  "I-ELIGIBILITY",
    "B-EVAL_CRITERIA","I-EVAL_CRITERIA",
]

LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL  = {i: label for label, i in LABEL2ID.items()}

BASE_MODEL = "dslim/bert-base-NER"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl(filepath):
    """
    Loads a JSON Lines annotation file.
    Each line: {"tokens": [...], "ner_tags": [...]}
    """
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} records from {filepath}")
    return records


def load_conll(filepath):
    """
    Loads a CoNLL-format file (token TAB label per line, blank lines between docs).
    Returns list of {"tokens": [...], "ner_tags": [...]} dicts.
    """
    records, current_tokens, current_labels = [], [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "":
                if current_tokens:
                    records.append({"tokens": current_tokens, "ner_tags": current_labels})
                    current_tokens, current_labels = [], []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    current_tokens.append(parts[0])
                    current_labels.append(parts[-1])
    if current_tokens:
        records.append({"tokens": current_tokens, "ner_tags": current_labels})
    logger.info(f"Loaded {len(records)} records from CoNLL file {filepath}")
    return records


def load_annotations(filepath):
    """Auto-detects format by extension and loads accordingly."""
    p = Path(filepath)
    if p.suffix in (".jsonl", ".json"):
        return load_jsonl(filepath)
    elif p.suffix in (".conll", ".txt", ".tsv"):
        return load_conll(filepath)
    else:
        try:
            return load_jsonl(filepath)
        except json.JSONDecodeError:
            return load_conll(filepath)


# ── Tokenization and label alignment ──────────────────────────────────────────

def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokenizes examples and aligns NER labels to BERT's WordPiece sub-tokens.

    Strategy: assign the word's label to its first sub-token; mark all
    subsequent sub-tokens with -100 so they are ignored by the loss function.
    This is the standard HuggingFace approach for token classification.
    """
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=512,
    )
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids    = tokenized.word_ids(batch_index=i)
        label_ids   = []
        prev_word   = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)                    # [CLS], [SEP], padding
            elif word_id != prev_word:
                label_ids.append(LABEL2ID.get(labels[word_id], 0))  # first sub-token
            else:
                label_ids.append(-100)                    # continuation sub-tokens
            prev_word = word_id
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized


# ── Evaluation ────────────────────────────────────────────────────────────────

def build_compute_metrics(label_list):
    """Returns a compute_metrics function compatible with HuggingFace Trainer."""
    seqeval = evaluate.load("seqeval")

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions    = np.argmax(logits, axis=-1)
        true_labels    = [[label_list[l] for l in row if l != -100] for row in labels]
        pred_labels    = [
            [label_list[p] for p, l in zip(prow, lrow) if l != -100]
            for prow, lrow in zip(predictions, labels)
        ]
        results = seqeval.compute(predictions=pred_labels, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall":    results["overall_recall"],
            "f1":        results["overall_f1"],
            "accuracy":  results["overall_accuracy"],
        }

    return compute_metrics


# ── Main training function ────────────────────────────────────────────────────

def fine_tune(train_file, eval_file, output_dir,
              epochs=5, batch_size=16, learning_rate=2e-5,
              base_model=BASE_MODEL):
    """
    Fine-tunes the BERT NER model on the annotated procurement dataset.

    Args:
        train_file:    Path to training annotations (JSONL or CoNLL).
        eval_file:     Path to evaluation annotations.
        output_dir:    Where to save the fine-tuned model checkpoint.
        epochs:        Training epochs.
        batch_size:    Per-device batch size.
        learning_rate: AdamW learning rate.
        base_model:    HuggingFace model ID to start from.
    """
    logger.info(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model     = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,   # resize the classification head for new labels
    )

    train_records = load_annotations(train_file)
    eval_records  = load_annotations(eval_file)
    dataset       = DatasetDict({
        "train":      Dataset.from_list(train_records),
        "validation": Dataset.from_list(eval_records),
    })

    logger.info("Tokenizing dataset and aligning labels...")
    tokenized = dataset.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=build_compute_metrics(LABEL_LIST),
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving fine-tuned model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    eval_results = trainer.evaluate()
    logger.info("Final evaluation:")
    for k, v in eval_results.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\nTraining complete. Model saved to: {output_dir}")
    print(f"Final F1: {eval_results.get('eval_f1', 'N/A'):.4f}")
    print(f"\nTo use the fine-tuned model:")
    print(f'  python document_processing_agent.py --model "{output_dir}" --data_dir ./tender_data')


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BERT NER on annotated tender data")
    parser.add_argument("--train_file",    required=True,
                        help="Training annotations (JSONL or CoNLL format)")
    parser.add_argument("--eval_file",     required=True,
                        help="Evaluation annotations")
    parser.add_argument("--output_dir",    default="./ner_model_finetuned")
    parser.add_argument("--epochs",        type=int,   default=5)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--base_model",    default=BASE_MODEL)
    args = parser.parse_args()

    fine_tune(
        train_file=args.train_file,
        eval_file=args.eval_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        base_model=args.base_model,
    )
