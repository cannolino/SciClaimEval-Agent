from openai import OpenAI
from datasets import load_dataset

api_key = '<api_key>'
base_url = "https://chat-ai.academiccloud.de/v1"

# available models with image processing capabilities
models = [
    "gemma-3-27b-it",
    "gemma-4-31b-it",
    "internvl3.5-30b-a3b",
    "mistral-large-3-675b-instruct-2512",
    "qwen3.5-122b-a10b",
    "qwen3.5-27b",
    "qwen3.5-35b-a3b",
    "qwen3.5-397b-a17b",
    "qwen3.6-35b-a3b",
    "qwen3-omni-30b-a3b-instruct"
]
model = "meta-llama-3.1-8b-instruct"
client = OpenAI(
    api_key = api_key,
    base_url = base_url
)
dataset = load_dataset("alabnii/sciclaimeval-shared-task", "task1")
dataset2 = load_dataset("alabnii/sciclaimeval-shared-task", "task2")

# TODO: add context (image) and adjust persona
def record_responses(dataset, dic, key, attributes):
    for data in dataset:
        # inject data into appropriate place
        chat_completion = client.chat.completions.create(
                messages= [
                    {"role":"system","content":"You are a scientific work reviewer"},
                    {"role":"user","content":"Does this provided evidence support or refute the given claim?"},
                ],
                model= model,
                max_completion_tokens= 1
            )
        # TODO: add the response to the dictionary using the key described, also specify attributes to be extracted for statistical purpose.
        print(chat_completion)
