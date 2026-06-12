"""
SciClaimEval - Task 1 - Multi-model pipeline (VLM extractor + reasoner).

Two-stage approach:

  Stage 1 (extraction, VLM):
    A vision-language model receives the evidence PNG (+ caption) and
    converts it to faithful text:
      - table  -> full Markdown transcription
      - figure -> detailed description (axes, legend, trends, values)
    The extraction does NOT see the claim, to avoid biasing the
    transcription. Results are cached on disk keyed by evi_path, so each
    image is extracted only once per extractor, even across re-runs or
    when new reasoners are added.

  Stage 2 (reasoning, any LLM):
    A text model receives claim + evi_type + caption + context + the
    extracted evidence, and answers Supported or Refuted. Text-only
    models work here because the image was already transcribed.

All combinations of EXTRACTOR_MODELS x models in phase_0/models_list.txt
are evaluated.

The only external parameter is --max-samples (0 = full dev set).

Outputs (in phase_0/results/):
  - cache_extractions_<extractor>.json             : extraction cache
  - predictions_task1_<extractor>__<reasoner>.json : official format
  - details_task1_<extractor>__<reasoner>.jsonl    : per-sample detail + errors
  - gold_task1_<n>.json                            : gold file for the subset
  - summary_task1_pipeline_<n>.csv                 : comparison of combinations

Evaluation uses the official script: evaluation/eval_claim.py
(eval_task_1_individual and eval_task_1_pair).
"""

import argparse
import base64
import csv
import importlib.util
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Silence per-file download bars from huggingface_hub. This must be set
# BEFORE importing the HF libraries, otherwise it is ignored.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from tqdm import tqdm

import datasets
from datasets import load_dataset
from huggingface_hub import hf_hub_download

# Silence dataset-loading progress bars and HF warnings (e.g. the
# "unauthenticated requests" notice). The console should only show our
# own global progress bar and the final summary.
try:
    datasets.disable_progress_bars()
except AttributeError:  # older datasets versions use a different API
    from datasets.utils.logging import disable_progress_bar
    disable_progress_bar()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)

# =============================================================================
# Fixed configuration.
# Everything is defined here by default; only --max-samples is external.
# =============================================================================

def find_project_root():
    """Walk upwards from this file until we find the folder that contains
    phase_0/models_list.txt. This makes the script work no matter which
    subfolder it is placed in (phase_0/, test/, ...)."""
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "phase_0" / "models_list.txt").exists():
            return candidate
    sys.exit("ERROR: phase_0/models_list.txt not found in the project.")


PROJECT_ROOT = find_project_root()

# KISSKI / Academic Cloud OpenAI-compatible endpoint.
BASE_URL = "https://chat-ai.academiccloud.de/v1"

# Dataset coordinates on Hugging Face.
DATASET_ID = "alabnii/sciclaimeval-shared-task"
DATASET_CONFIG = "task1"
DATASET_SPLIT = "dev"

# Project paths (all relative to the project root).
MODELS_FILE = PROJECT_ROOT / "phase_0" / "models_list.txt"
OUTPUT_DIR = PROJECT_ROOT / "phase_0" / "results"
EVAL_SCRIPT = PROJECT_ROOT / "evaluation" / "eval_claim.py"

# VLMs used as extractors in stage 1. Edit this list to try other ones.
# Every model here must support image input.
EXTRACTOR_MODELS = [
    "qwen3-omni-30b-a3b-instruct",
    "internvl3.5-30b-a3b",
]

VALID_LABELS = ("Supported", "Refuted")

# API call parameters.
EXTRACTION_MAX_TOKENS = 2048   # large tables need room to be transcribed
REASONING_MAX_TOKENS = 4096    # reasoning models spend tokens thinking before
                               # they answer; with a small budget they hit the
                               # token limit mid-thought and return nothing
TEMPERATURE = 0                # deterministic outputs for reproducibility
API_RETRIES = 5                # retries on transient API errors
RETRY_WAIT_S = 3               # base wait for generic errors (3,6,12,24,48s)
RATE_LIMIT_WAIT_S = 15         # base wait for 429s (15,30,60,120,240s):
                               # KISSKI's quota needs real time to clear,
                               # short retries just burn more quota
API_TIMEOUT_S = 120            # abort hung calls (model down or queued)
MAX_CONCURRENT_REQUESTS = 2    # parallel API calls; KISSKI rate-limits
                               # aggressively (confirmed 429s at 8-16)


# =============================================================================
# Official evaluation.
# evaluation/eval_claim.py is imported as-is so the metrics are computed
# exactly like the shared task does (no duplicated code here).
# =============================================================================

def load_official_evaluator():
    """Import eval_claim.py by file path and return the module object."""
    spec = importlib.util.spec_from_file_location("eval_claim", EVAL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =============================================================================
# API client.
# The key is loaded from .env (KISSKI_API_KEY), never hardcoded.
# =============================================================================

def create_client():
    """Create the OpenAI client pointing at KISSKI, with the key from .env."""
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv()  # fallback: .env in the current working directory

    api_key = os.getenv("KISSKI_API_KEY")
    if not api_key:
        sys.exit("ERROR: KISSKI_API_KEY missing from .env")

    # timeout: prevents the run from hanging forever if a model is down.
    # max_retries=0: retries are handled by call_model(), not by the SDK.
    return OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        timeout=API_TIMEOUT_S,
        max_retries=0,
    )


# =============================================================================
# Dataset loading and gold file.
# =============================================================================

def load_dev_set(max_samples):
    """Load the Task 1 dev split. max_samples=0 means the full split."""
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT)

    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    return dataset


def save_gold_file(path, dev_set):
    """Write the gold file with exactly the fields the official evaluator
    needs (claim_id, claim_id_pair, label) for the evaluated subset."""
    rows = [
        {
            "claim_id": ex["claim_id"],
            "claim_id_pair": ex.get("claim_id_pair"),
            "label": ex["label"],
        }
        for ex in dev_set
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


# =============================================================================
# Evidence image handling.
#
# evi_path looks like "tables_png/dev/val_tab_0001.png", but in the HF repo
# the files live under "data/", so the real path is
# "data/tables_png/dev/val_tab_0001.png". The "data/" prefix is mandatory.
# =============================================================================

_image_cache = {}  # in-memory cache: evi_path -> base64 data URL


def get_image_data_url(evi_path):
    """Download the image from the HF repo (hf_hub_download keeps a local
    file cache) and return it as a base64 data URL, the format expected
    by OpenAI-compatible APIs for image input."""
    if evi_path in _image_cache:
        return _image_cache[evi_path]

    local_path = hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        filename=f"data/{evi_path}",   # <- mandatory "data/" prefix
    )

    with open(local_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    data_url = f"data:image/png;base64,{encoded}"
    _image_cache[evi_path] = data_url
    return data_url


# =============================================================================
# Generic API call with retries.
# =============================================================================

def call_model(client, model, messages, max_tokens):
    """Call the API, retrying on transient errors with exponential
    backoff (3, 6, 12, 24 seconds). Rate-limit errors (429) are the main
    target: a fixed short wait is not enough when the server is saturated.
    Raises the last error if all attempts fail (the caller records it)."""
    last_error = None
    for attempt in range(API_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=max_tokens,
            )
            choice = response.choices[0]
            message = choice.message
            content = message.content or ""

            # Some reasoning models leave content empty and put the text
            # in another field; try the known alternatives.
            if not content.strip():
                for field in ("reasoning_content", "reasoning"):
                    content = getattr(message, field, "") or ""
                    if content.strip():
                        break

            # Still empty: return a debug dump of the full message plus
            # finish_reason, so raw_output in the details JSONL shows WHY
            # the model produced nothing (e.g. finish_reason="length"
            # means the thinking budget ate all the tokens).
            if not content.strip():
                return (
                    f"[EMPTY RESPONSE] finish_reason={choice.finish_reason} "
                    f"message={message.model_dump_json()}"
                )

            return content.strip()
        except RateLimitError as e:
            # 429: the quota is exhausted. Retrying fast only makes it
            # worse; wait long enough for the window to clear.
            last_error = e
            if attempt < API_RETRIES:
                time.sleep(RATE_LIMIT_WAIT_S * (2 ** attempt))
        except Exception as e:
            last_error = e
            if attempt < API_RETRIES:
                time.sleep(RETRY_WAIT_S * (2 ** attempt))

    raise last_error


# =============================================================================
# Stage 1: extract the visual evidence to text (VLM).
# =============================================================================

def build_extraction_prompt(example):
    """Build the extraction prompt. The claim is deliberately excluded so
    the transcription cannot be biased towards supporting/refuting it."""
    if example["evi_type"] == "table":
        instruction = (
            "Transcribe the table in this image to Markdown format. "
            "Reproduce it EXACTLY: every row, every column, every header, "
            "every number and symbol (bold/blue markers can be noted in "
            "parentheses). Do not summarize, do not omit rows, do not "
            "round numbers. Output only the Markdown table."
        )
    else:
        instruction = (
            "Describe this scientific figure in exhaustive detail: type of "
            "plot, axes (names, units, ranges), legend entries, each curve "
            "or bar and its approximate values, visible trends, crossings, "
            "maxima/minima, and any annotations. Be precise with numbers. "
            "Do not interpret conclusions; only describe what is shown."
        )

    return (
        f"{instruction}\n\n"
        f"For reference, the original caption is:\n{example['caption']}"
    )


def load_extraction_cache(extractor):
    """Load the on-disk extraction cache for this extractor, if any."""
    path = OUTPUT_DIR / f"cache_extractions_{safe_name(extractor)}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_extraction_cache(extractor, cache):
    """Persist the extraction cache to disk."""
    path = OUTPUT_DIR / f"cache_extractions_{safe_name(extractor)}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def run_extraction_stage(client, extractor, dev_set, pbar):
    """Extract every evidence image in the subset with one extractor.

    Returns a dict: evi_path -> {"text": ..., "error": ...}.

    The disk cache means repeated runs (or adding reasoners later) do not
    pay for extraction again. Entries that previously failed ARE retried.

    pbar is the single global progress bar; it advances one unit per
    sample (cache hits advance instantly).

    Pending images are extracted in parallel (MAX_CONCURRENT_REQUESTS)."""
    cache = load_extraction_cache(extractor)

    # Collect the unique images that still need extraction. Cache hits
    # (and duplicate evi_paths) only advance the progress bar.
    pending = {}
    for example in dev_set:
        evi_path = example["evi_path"]
        if evi_path in cache and not cache[evi_path].get("error"):
            pbar.update(1)  # cache hit from a previous run
        elif evi_path in pending:
            pbar.update(1)  # duplicate image within this subset
        else:
            pending[evi_path] = example

    def extract_one(example):
        """Extract a single image. Never raises: errors are recorded in
        the returned entry so the rest of the batch continues."""
        entry = {"text": "", "error": ""}
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_extraction_prompt(example)},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": get_image_data_url(example["evi_path"])
                            },
                        },
                    ],
                }
            ]
            entry["text"] = call_model(
                client, extractor, messages, EXTRACTION_MAX_TOKENS
            )
            if not entry["text"]:
                entry["error"] = "empty extraction"
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
        return entry

    # Run extractions in parallel. Results are written to the cache from
    # this (single) thread as they complete, so no locking is needed.
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        futures = {
            executor.submit(extract_one, example): evi_path
            for evi_path, example in pending.items()
        }
        for future in as_completed(futures):
            cache[futures[future]] = future.result()
            completed += 1
            pbar.update(1)

            # Incremental save: if the run is interrupted, the cache survives.
            if completed % 10 == 0:
                save_extraction_cache(extractor, cache)

    save_extraction_cache(extractor, cache)
    return cache


# =============================================================================
# Stage 2: reason over the extracted evidence (pure text).
# =============================================================================

def build_reasoning_prompt(example, evidence_text):
    """Build the reasoning prompt: claim + metadata + extracted evidence."""
    return (
        "You are a scientific claim verification assistant.\n"
        "The visual evidence (a table or figure from a paper) has been "
        "transcribed to text below. Decide whether it supports or refutes "
        "the claim.\n\n"
        f"Claim:\n{example['claim']}\n\n"
        f"Evidence type:\n{example['evi_type']}\n\n"
        f"Evidence caption:\n{example['caption']}\n\n"
        f"Context:\n{example['context']}\n\n"
        f"Transcribed evidence:\n{evidence_text}\n\n"
        "Answer with exactly one word: Supported or Refuted."
    )


def normalize_prediction(raw_output):
    """Map raw model output to Supported / Refuted / Invalid.

    Invalid means no usable label could be recovered; check raw_output
    and error in the details JSONL to see what the model actually said."""
    if not raw_output:
        return "Invalid"

    text = raw_output.strip().lower()

    # Ideal case: the model answered with exactly one word.
    first_word = re.sub(r"[^a-z]", "", text.split()[0]) if text.split() else ""
    if first_word in ("supported", "support", "supports"):
        return "Supported"
    if first_word in ("refuted", "refute", "refutes"):
        return "Refuted"

    # Negations like "does not support" -> Refuted.
    if re.search(r"\bnot\s+support", text):
        return "Refuted"

    # Otherwise scan the whole text: the first mention wins.
    sup = re.search(r"\bsupport(?:s|ed)?\b", text)
    ref = re.search(r"\b(?:refut(?:e|es|ed)|contradict\w*)\b", text)

    if sup and ref:
        return "Supported" if sup.start() < ref.start() else "Refuted"
    if sup:
        return "Supported"
    if ref:
        return "Refuted"

    return "Invalid"


def run_reasoning_stage(client, reasoner, dev_set, extractions, pbar):
    """Evaluate one reasoner over the already-extracted evidence.

    If the extraction failed for a sample, the sample is marked Invalid
    with the real error recorded, and the reasoner is not called. If a
    single sample fails, the others are unaffected.

    pbar is the single global progress bar (one unit per sample).
    Samples are processed in parallel (MAX_CONCURRENT_REQUESTS); the
    results list preserves the dataset order."""

    def reason_one(example):
        """Process a single sample. Never raises: errors are recorded in
        the returned record so the rest of the batch continues."""
        extraction = extractions.get(example["evi_path"], {})

        record = {
            "claim_id": example["claim_id"],
            "claim": example["claim"],
            "caption": example["caption"],
            "context": example["context"],
            "evi_type": example["evi_type"],
            "evi_path": example["evi_path"],
            "gold_label": example["label"],
            "evidence_text": extraction.get("text", ""),
            "pred_label": "Invalid",
            "raw_output": "",
            "error": "",
        }

        if extraction.get("error") or not extraction.get("text"):
            record["error"] = (
                f"extraction failed: {extraction.get('error', 'missing')}"
            )
            return record

        try:
            messages = [
                {
                    "role": "user",
                    "content": build_reasoning_prompt(example, extraction["text"]),
                }
            ]
            raw_output = call_model(
                client, reasoner, messages, REASONING_MAX_TOKENS
            )
            record["raw_output"] = raw_output
            record["pred_label"] = normalize_prediction(raw_output)
        except Exception as e:
            record["error"] = f"{type(e).__name__}: {e}"

        return record

    # Parallel execution, keeping results aligned with dataset order.
    results = [None] * len(dev_set)
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        futures = {
            executor.submit(reason_one, example): i
            for i, example in enumerate(dev_set)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            pbar.update(1)

    return results


# =============================================================================
# Output files.
# =============================================================================

def safe_name(model):
    """Turn a model name into a filename-safe string."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", model)


def save_predictions(path, results):
    """Official prediction JSON: only claim_id and pred_label."""
    preds = [{"claim_id": r["claim_id"], "pred_label": r["pred_label"]} for r in results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)


def save_details(path, results):
    """Detailed JSONL, one sample per line. This is the file to inspect
    when a sample comes out Invalid (raw_output, evidence_text, error)."""
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


SUMMARY_FIELDS = [
    "extractor", "reasoner", "samples",
    "precision", "recall", "macro_f1", "accuracy",
    "pair_accuracy", "correct_pairs", "total_pairs",
    "invalid_predictions", "sample_errors", "error",
]


def save_summary(path, rows):
    """Comparison CSV: one row per extractor x reasoner combination."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary_table(rows):
    """Print the final summary as an aligned table in the console.

    This is the same content as the CSV, shown directly so results can
    be read without opening the file."""
    # Compute the width of each column from header and values.
    widths = {
        field: max(len(field), *(len(str(row.get(field, ""))) for row in rows))
        for field in SUMMARY_FIELDS
    }

    header = " | ".join(field.ljust(widths[field]) for field in SUMMARY_FIELDS)
    separator = "-+-".join("-" * widths[field] for field in SUMMARY_FIELDS)

    print("\n" + header)
    print(separator)
    for row in rows:
        print(
            " | ".join(
                str(row.get(field, "")).ljust(widths[field])
                for field in SUMMARY_FIELDS
            )
        )


# =============================================================================
# Main.
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SciClaimEval Task 1: VLM extractor + reasoner pipeline."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Number of dev-set samples to evaluate (0 = all).",
    )
    args = parser.parse_args()

    eval_claim = load_official_evaluator()
    client = create_client()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reasoner list (empty lines and # comments are ignored).
    reasoners = [
        line.strip()
        for line in MODELS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    dev_set = load_dev_set(args.max_samples)
    n = len(dev_set)

    # Gold file for the exact subset being evaluated.
    gold_path = OUTPUT_DIR / f"gold_task1_{n}.json"
    save_gold_file(gold_path, dev_set)

    summary_rows = []
    summary_path = OUTPUT_DIR / f"summary_task1_pipeline_{n}.csv"

    # One single progress bar for the WHOLE run. Each unit is one model
    # call (extraction or reasoning), so the ETA reflects total runtime:
    #   total = extractors*n  (stage 1)  +  extractors*reasoners*n  (stage 2)
    total_units = len(EXTRACTOR_MODELS) * n * (1 + len(reasoners))
    pbar = tqdm(total=total_units, desc="Evaluating", unit="call")

    for extractor in EXTRACTOR_MODELS:
        # If a whole extractor fails (model down), record an error row for
        # each pending reasoner and continue with the next extractor.
        try:
            extractions = run_extraction_stage(client, extractor, dev_set, pbar)
        except Exception as e:
            error_msg = f"extractor failed: {type(e).__name__}: {e}"
            for reasoner in reasoners:
                row = {field: "" for field in SUMMARY_FIELDS}
                row.update(
                    {
                        "extractor": extractor,
                        "reasoner": reasoner,
                        "samples": n,
                        "error": error_msg,
                    }
                )
                summary_rows.append(row)
            save_summary(summary_path, summary_rows)
            # Skip the bar forward over this extractor's pending work.
            pbar.update(n * len(reasoners))
            continue

        for reasoner in reasoners:
            row = {field: "" for field in SUMMARY_FIELDS}
            row.update(
                {"extractor": extractor, "reasoner": reasoner, "samples": n}
            )

            try:
                results = run_reasoning_stage(
                    client, reasoner, dev_set, extractions, pbar
                )

                combo = f"{safe_name(extractor)}__{safe_name(reasoner)}"
                pred_path = OUTPUT_DIR / f"predictions_task1_{combo}.json"
                details_path = OUTPUT_DIR / f"details_task1_{combo}.jsonl"
                save_predictions(pred_path, results)
                save_details(details_path, results)

                # Official evaluation: individual metrics + pair accuracy.
                individual = eval_claim.eval_task_1_individual(pred_path, gold_path)
                pair = eval_claim.eval_task_1_pair(pred_path, gold_path)

                row.update(individual)
                row.update(pair)
                row.pop("confusion_matrix", None)  # not a CSV column
                row["invalid_predictions"] = sum(
                    1 for r in results if r["pred_label"] == "Invalid"
                )
                row["sample_errors"] = sum(1 for r in results if r["error"])

            except Exception as e:
                # If a reasoner fails entirely, record it and move on.
                row["error"] = f"{type(e).__name__}: {e}"

            summary_rows.append(row)

            # Rewrite the summary after every combination: if the run is
            # interrupted, completed results are not lost.
            save_summary(summary_path, summary_rows)

    pbar.close()

    # Final results: same content as the CSV, printed to the console.
    print_summary_table(summary_rows)
    print("\nSummary saved to:", summary_path)


if __name__ == "__main__":
    main()