from transformers import AutoModelForImageTextToText, AutoModelForMultimodalLM, AutoProcessor, LogitsProcessor
from pathlib import Path
import argparse, json, torch

class LogitsMask(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = set(allowed_token_ids)
    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float('-inf'))
        for token_id in self.allowed_token_ids:
            mask[:, token_id] = 0
        scores = scores + mask
        return scores

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate scientific claims using LLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '-i', '--input-file',
        type=str,
        required=True,
        help='Path to the input file (JSON format)'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        required=True,
        help='Path to the model directory'
    )
    parser.add_argument(
        '-o', '--output-file',
        type=str,
        required=True,
        help='Path to the output file where results will be saved'
    )
    args = parser.parse_args()
    input_file = Path(args.input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    model = Path(args.model)
    if not model.exists():
        raise FileNotFoundError(f"Model not found: {model}")
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_responses(input_file, model, output_file)

def getMask(processor):
    tokenizer = processor.tokenizer
    allowed_token_ids = []
    banned_tokens = ["<bos>"]
    for token in ["true", "false"]:
        token_ids = tokenizer.encode(token)
        if type(token_ids) is list:
            for token_id in token_ids:
                if processor.decode(token_id) not in banned_tokens:
                    allowed_token_ids.append(token_id)
        else:
            allowed_token_ids.append(token_ids)
    print(f"Allowed token IDs: {allowed_token_ids}\n")
    return LogitsMask(allowed_token_ids)

def load_model(model_path):
    if 'gemma' in str(model_path):
        model = AutoModelForImageTextToText.from_pretrained(model_path, device_map="auto", dtype="auto")
        processor = AutoProcessor.from_pretrained(model_path, padding_side='left')
    if 'Qwen' in str(model_path):
        model = AutoModelForMultimodalLM.from_pretrained(model_path, device_map="auto", dtype="auto")
        processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

def write_responses(input_file, model_path, output_file):
    model, processor = load_model(model_path)
    tokenizer = processor.tokenizer
    logits_mask = getMask(processor)
    ls = list()
    with open(input_file, "r") as f:
        dataset = json.load(f)
    for data in dataset:
        if data["use_context"] == "no":
            prompt = 'The claim: "' + data["claim"] + '" is supported, given evidence depicted in the image.'
        else:
            prompt = 'Given following context: "' + data["context"] + '" and evidence depicted in the image, the claim: "' \
                + data["claim"] + '" is supported.'
        # TODO: if data["use_context"] == "other sources":
        messages = [
            {
                "role": "system", "content": "You are a scientific reviewer.",
                "role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "url": str(input_file.parent / data["evi_path"])},
                ]
            },
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=model.dtype)
        input_len = inputs["input_ids"].shape[-1]

        output = model.generate(**inputs, max_new_tokens=1, logits_processor=[logits_mask])
        if processor.decode(output[0][input_len:], skip_special_tokens=True, cache_implementation='static') == "true":
            ls.append({"claim_id": data["claim_id"], "pred_label": 'Supported'})
        else:
            ls.append({"claim_id": data["claim_id"], "pred_label": 'Refuted'})

    with open(output_file, "w") as f:
        json.dump(ls, f, indent=2)

if __name__ == "__main__":
    main()
