import os
import json
import argparse
from pathlib import Path

from openai import OpenAI
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score


BASE_URL = "https://chat-ai.academiccloud.de/v1"
DEFAULT_MODEL = "meta-llama-3.1-8b-instruct"


def build_prompt(example):
    return f"""
You are a scientific claim verification assistant.

Claim:
{example["claim"]}

Evidence type:
{example["evi_type"]}

Evidence caption:
{example["caption"]}

Context:
{example["context"]}

Question:
Does the evidence support or refute the claim?

Answer only with one word:
Supported or Refuted
""".strip()


def normalize_prediction(text):
    text = text.strip().lower()

    if "supported" in text:
        return "Supported"

    if "refuted" in text:
        return "Refuted"

    return "Invalid"


def call_model(client, model, prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a scientific reviewer."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_completion_tokens=16,
    )

    return response.choices[0].message.content.strip()


def evaluate_model(client, model, max_samples):
    dataset = load_dataset("alabnii/sciclaimeval-shared-task", "task1")
    dev_set = dataset["dev"].select(range(max_samples))

    results = []

    for i, example in enumerate(dev_set):
        prompt = build_prompt(example)
        raw_output = call_model(client, model, prompt)

        pred_label = normalize_prediction(raw_output)
        gold_label = example["label"]

        result = {
            "index": i,
            "paper_id": example["paper_id"],
            "claim_id": example["claim_id"],
            "claim": example["claim"],
            "gold_label": gold_label,
            "pred_label": pred_label,
            "raw_output": raw_output,
            "correct": pred_label == gold_label,
        }

        results.append(result)

        print(f"[{i + 1}/{max_samples}] {example['claim_id']} | gold={gold_label} | pred={pred_label}")

    return results


def compute_metrics(results):
    gold = [r["gold_label"] for r in results]
    pred = [r["pred_label"] for r in results]

    valid_gold = []
    valid_pred = []

    invalid_count = 0

    for g, p in zip(gold, pred):
        if p == "Invalid":
            invalid_count += 1
        else:
            valid_gold.append(g)
            valid_pred.append(p)

    accuracy = accuracy_score(gold, pred)
    macro_f1 = f1_score(
        valid_gold,
        valid_pred,
        labels=["Supported", "Refuted"],
        average="macro",
        zero_division=0,
    )

    return {
        "samples": len(results),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "invalid_predictions": invalid_count,
    }


def save_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_summary(path, model, metrics):
    with open(path, "w", encoding="utf-8") as f:
        f.write("model,samples,accuracy,macro_f1,invalid_predictions\n")
        f.write(
            f"{model},{metrics['samples']},{metrics['accuracy']},"
            f"{metrics['macro_f1']},{metrics['invalid_predictions']}\n"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--output-dir", default="phase_0/results")

    args = parser.parse_args()

    api_key = ""

    if not api_key:
        raise ValueError("Missing KISSKI_API_KEY environment variable")

    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate_model(
        client=client,
        model=args.model,
        max_samples=args.max_samples,
    )

    metrics = compute_metrics(results)

    predictions_path = output_dir / "predictions_text.jsonl"
    summary_path = output_dir / "summary_text.csv"

    save_jsonl(predictions_path, results)
    save_summary(summary_path, args.model, metrics)

    print("\nFINAL RESULTS")
    print("Model:", args.model)
    print("Samples:", metrics["samples"])
    print("Accuracy:", metrics["accuracy"])
    print("Macro-F1:", metrics["macro_f1"])
    print("Invalid predictions:", metrics["invalid_predictions"])

    print("\nSaved:")
    print(predictions_path)
    print(summary_path)


if __name__ == "__main__":
    main()