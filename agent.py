from transformers import AutoModelForImageTextToText, AutoModelForMultimodalLM, AutoProcessor, LogitsProcessor
from transformers import pipeline
from pathlib import Path
import json, torch, weave

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
    def __init__(self, model):
        self.model = None
        self.processor = None
        self.sentiment_model = None
        self.logits_mask = None

        if not Path(model).exists():
            raise FileNotFoundError(f"Model not found: {model}")
        self._load_model(model)

    def _load_model(self, model):
        """Load the main model and create corresponding Logits Mask"""
        if 'gemma' in model:
            self.model = AutoModelForImageTextToText.from_pretrained(
                Path(model), device_map="auto", dtype="auto", attn_implementation="sdpa"
            )
            self.processor = AutoProcessor.from_pretrained(
                Path(model), padding_side='left'
            )
            weave.init('sciclaimeval/base-gemma')
        elif 'Qwen' in model:
            self.model = AutoModelForMultimodalLM.from_pretrained(
                Path(model), device_map="auto", dtype="auto"
            )
            self.processor = AutoProcessor.from_pretrained(Path(model))
            weave.init('sciclaimeval/base-qwen')
        else:
            raise ValueError(f"Unsupported model: {model}")

        self.logits_mask = self._create_logits_mask()

    def _load_sentiment_model(self, sentiment_model):
        """Load the optional sentiment model"""
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model=sentiment_model
        )

    def _create_logits_mask(self):
        """Create logits mask for binary classification"""
        allowed_token_ids = []
        banned_tokens = ["<bos>"]

        for token in ["true", "false"]:
            token_ids = self.processor.tokenizer.encode(token)
            if isinstance(token_ids, list):
                for token_id in token_ids:
                    if self.processor.decode(token_id) not in banned_tokens:
                        allowed_token_ids.append(token_id)
            else:
                allowed_token_ids.append(token_ids)

        # print(f"Allowed token IDs: {allowed_token_ids}\n")
        return LogitsMask(allowed_token_ids)

    def _create_prompt(self, data_item, use_sentiment=False):
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
        dic["reasoning"] = inputs["reasoning_text"]
        return dic

    @weave.op(postprocess_inputs=postprocess_inputs)
    def _classify_with_sentiment(self, data_item, reasoning_text):
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
        prompt = self._create_prompt(data_item)

        messages = [
            {
                "role": "system",
                "content": "You are a scientific reviewer."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "url": str(input_file.parent / data_item["evi_path"])}
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
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
                "content": "You are a scientific reviewer."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "url": str(input_file.parent / data_item["evi_path"])}
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
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
        pred_label = self._classify_with_sentiment(data_item, self._generate_reasoning(data_item, input_file))
        return {
            "claim_id": data_item["claim_id"],
            "pred_label": pred_label
        }

    # TODO: evaluate this (after fine-tuning using SPIQA dataset) also log to wandb
    @weave.op()
    def _generate_verification_questions(self, data_item, input_file, num_questions):
        """
        Transform a claim into verification questions that can help evaluate its validity

        Args:
            data_item: Dictionary containing claim information
            input_file: Path to the input file containing the dataset
            num_questions: Number of verification questions to generate (default: 3)

        Returns:
            List of verification questions
        """
        if data_item["use_context"] == "no":
            prompt = f'Generate {num_questions} verification questions that would help determine if the claim "{data_item["claim"]}" is supported by the evidence in the image. Each question should be concise and directly relevant to verifying the claim.'
        else:
            prompt = f'Generate {num_questions} verification questions that would help determine if the claim "{data_item["claim"]}" is supported by the context "{data_item["context"]}" and the evidence in the image. Each question should be concise and directly relevant to verifying the claim.'

        messages = [
            {
                "role": "system",
                "content": "You are a scientific reviewer. You are currently discussing a couple of questions that are related to the paper with your colleague."
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
            return_tensors="pt"
        ).to(self.model.device, dtype=self.model.dtype)

        output = self.model.generate(
            **inputs,
            max_new_tokens=512
        )

        response = self.processor.decode(
            output[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
            cache_implementation='static'
        )

        # Parse the response into individual questions
        # questions = []
        # for line in response.split('\n'):
        #     line = line.strip()
        #     if line and line.endswith('?'):
        #         questions.append(line)
        #     if len(questions) >= num_questions:
        #         break

        # return questions[:num_questions]
        return response

    def generate_questions(self, input_file, output_file, num_questions):
        """Main method to evaluate claims from input file and save results"""
        input_file = Path(input_file)
        output_file = Path(output_file)
        results = []

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(input_file, "r") as f:
            dataset = json.load(f)

        for data_item in dataset:
            result = self._generate_verification_questions(data_item, input_file, num_questions)
            results.append(result)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    def evaluate_claims(self, input_file, output_file, sentiment_model=None):
        """Main method to evaluate claims from input file and save results"""
        input_file = Path(input_file)
        output_file = Path(output_file)
        results = []

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(input_file, "r") as f:
            dataset = json.load(f)

        if sentiment_model:
            sentiment_model = Path(sentiment_model)
            if not sentiment_model.exists():
                raise FileNotFoundError(f"Sentiment Model not found: {sentiment_model}")
            self._load_sentiment_model(sentiment_model)

        for data_item in dataset:
            result = self._process_single_claim(data_item, input_file)
            results.append(result)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)