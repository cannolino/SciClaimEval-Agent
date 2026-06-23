from transformers import AutoModelForImageTextToText, AutoModelForMultimodalLM, AutoProcessor, LogitsProcessor
from transformers import pipeline
from pathlib import Path
from PIL import Image
import argparse, json, torch, weave

class LogitsMask(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float('-inf'))
        for token_id in self.allowed_token_ids:
            mask[:, token_id] = 0
        scores = scores + mask
        return scores

class Agent:
    def __init__(self, model_path, sentiment_model_path):
        self.model_path = Path(model_path)
        self.sentiment_model_path = Path(sentiment_model_path) if sentiment_model_path else None
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.sentiment_model = None
        self.logits_mask = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.sentiment_model_path.exists():
            raise FileNotFoundError(f"Sentiment Model not found: {self.sentiment_model_path}")

        self._load_models()

    def _load_models(self):
        """Load the main model and optional sentiment analysis model"""
        if 'gemma' in str(self.model_path):
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path, device_map="auto", dtype="auto", attn_implementation="sdpa"
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, padding_side='left'
            )
            weave.init('sciclaimeval/base-gemma')
        elif 'Qwen' in str(self.model_path):
            self.model = AutoModelForMultimodalLM.from_pretrained(
                self.model_path, device_map="auto", dtype="auto"
            )
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            weave.init('sciclaimeval/base-qwen')
        else:
            raise ValueError(f"Unsupported model type in path: {self.model_path}")

        self.tokenizer = self.processor.tokenizer

        if self.sentiment_model_path:
            self.sentiment_model = pipeline(
                "sentiment-analysis",
                model=self.sentiment_model_path
            )

        if not self.sentiment_model:
            self.logits_mask = self._create_logits_mask()

    def _create_logits_mask(self):
        """Create logits mask for binary classification"""
        allowed_token_ids = []
        banned_tokens = ["<bos>"]

        for token in ["true", "false"]:
            token_ids = self.tokenizer.encode(token)
            if isinstance(token_ids, list):
                for token_id in token_ids:
                    if self.processor.decode(token_id) not in banned_tokens:
                        allowed_token_ids.append(token_id)
            else:
                allowed_token_ids.append(token_ids)

        # print(f"Allowed token IDs: {allowed_token_ids}\n")
        return LogitsMask(allowed_token_ids)

    def _create_prompt(self, data_item, use_sentiment):
        """Create prompt based on context usage and whether sentiment analysis is used"""
        if use_sentiment:
            # Prompt for reasoning when using sentiment analysis
            if data_item["use_context"] == "no":
                return f'Based on the evidence depicted in the image, is the claim: "{data_item["claim"]}" supported? Provide short explanation for your conclusion.'
            # TODO: add prompt for other sources
            else:
                return f'Given the context: "{data_item["context"]}" and the evidence depicted in the image, is the claim: "{data_item["claim"]}" supported? Provide short explanation for your conclusion.'
        else:
            if data_item["use_context"] == "no":
                return f'The claim: "{data_item["claim"]}" is supported, given evidence depicted in the image.'
            # TODO: add prompt for other sources
            else:
                return f'Given following context: "{data_item["context"]}" and evidence depicted in the image, the claim: "{data_item["claim"]}" is supported.'

    def postprocess_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
        dic = dict()
        dic["evi_path"] = inputs["data_item"]["evi_path"]
        dic["claim"] = inputs["data_item"]["claim"]
        dic["img_size"] = Image.open(inputs["input_file"].parent / inputs["data_item"]["evi_path"]).size
        dic["reasoning"] = inputs["reasoning_text"]
        return dic

    @weave.op(postprocess_inputs=postprocess_inputs)
    def _classify_with_sentiment(self, data_item, input_file, reasoning_text):
        """Classify reasoning text using sentiment analysis model"""
        sentiment_result = self.sentiment_model(reasoning_text)
        label = sentiment_result[0]['label']
        if label == 'POSITIVE':
            return 'Supported'
        return 'Refuted'

    def _process_single_claim(self, data_item, input_file):
        """Process a single claim and return prediction"""
        if self.sentiment_model:
            return self._process_claim_with_sentiment(data_item, input_file)
        return self._process_claim_direct(data_item, input_file)

    def _process_claim_direct(self, data_item, input_file):
        """Process claim using direct binary classification (original approach)"""
        prompt = self._create_prompt(data_item, use_sentiment=False)

        messages = [
            {
                "role": "system",
                "content": "You are a scientific reviewer."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "url": str(input_file.parent / data_item["evi_path"])},
                ]
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=self.model.dtype)
        input_len = inputs["input_ids"].shape[-1]

        output = self.model.generate(
            **inputs,
            max_new_tokens=1,
            logits_processor=[self.logits_mask]
        )

        prediction = self.processor.decode(
            output[0][input_len:],
            skip_special_tokens=True,
            cache_implementation='static'
        )

        return {
            "claim_id": data_item["claim_id"],
            "pred_label": 'Supported' if prediction == "true" else 'Refuted'
        }

    def _generate_reasoning(self, data_item, input_file):    
        prompt = self._create_prompt(data_item, use_sentiment=True)

        messages = [
            {
                "role": "system",
                "content": "You are a scientific reviewer. Provide detailed reasoning for your analysis."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "url": str(input_file.parent / data_item["evi_path"])},
                ]
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=self.model.dtype)

        output = self.model.generate(
            **inputs,
            max_new_tokens=420
        )
        
        return self.processor.decode(
            output[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
            cache_implementation='static'
        )

    def _process_claim_with_sentiment(self, data_item, input_file):
        """Process claim using reasoning + sentiment analysis approach"""
        pred_label = self._classify_with_sentiment(data_item, input_file, self._generate_reasoning(data_item, input_file))
        return {
            "claim_id": data_item["claim_id"],
            "pred_label": pred_label
        }

    def evaluate_claims(self, input_file, output_file):
        """Main method to evaluate claims from input file and save results"""
        input_file = Path(input_file)
        output_file = Path(output_file)
        results = []

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(input_file, "r") as f:
            dataset = json.load(f)

        for data_item in dataset[:20]:
            result = self._process_single_claim(data_item, input_file)
            results.append(result)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
