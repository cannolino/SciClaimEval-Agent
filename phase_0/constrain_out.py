from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LogitsProcessor
import torch

model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

class LogitsMask(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = set(allowed_token_ids)
    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float('-inf'))
        for token_id in self.allowed_token_ids:
            mask[:, token_id] = 0
        scores = scores + mask
        return scores

allowed_token_ids = []
for token in ["Supported", "Refuted"]:
    token_id = tokenizer.encode(token)
    allowed_token_ids.append(token_id[0])
    print(f"First token of '{token}': {token_id} -> {tokenizer.decode(token_id)}")
print(f"Allowed token IDs: {allowed_token_ids}\n")

lm_test = LogitsMask(allowed_token_ids)

prompt = ["Does the claim: 'I am a vegan' support or refute the claim: 'I eat vegetable'?"]
model_inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
model_output = model.generate(
    **model_inputs,
    max_new_tokens=1,
    logits_processor=[lm_test],
)

print(f"Input: {prompt[0]}")
print(f"Output: {tokenizer.decode(model_output[0], skip_special_tokens=True).replace(prompt[0], "")}")