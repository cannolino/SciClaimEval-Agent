from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from PIL import Image
import random, wandb, torch

PROJECT = "gemma-4-31B-naive"

base_config = {
    "base_model": "google/gemma-4-31B",
    "processor": "google/gemma-4-31B-it",
    "task": "SciClaimEval",
    "seed": 42,
    "precision": "lora-bf16",
    "learning_rate": 1e-5,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
}
random.seed(base_config["seed"])
run = wandb.init(
    entity='sciclaimeval',
    project=PROJECT,
    config=base_config,
    tags=["gemma4", "scientific", "vlm"],
)
print(f"Run: {run.name}")

processor = AutoProcessor.from_pretrained("models/gemma-4-31B")

def format_data(sample):
    if sample["use_context"] == "no":
        prompt = f'The claim: "{sample["claim"]}" is supported, given evidence depicted in the image.'
    # TODO: add prompt for other sources
    else:
        prompt = f'Given following context: "{sample["context"]}" and evidence depicted in the image, the claim: "{sample["claim"]}" is supported.'
    return {
        "messages" : [
            {
                "role": "system",
                "content": "You are a scientific reviewer."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": Image.open('dataset/' + sample["evi_path"])}
                ]
            },
            {
                "role": "assistant",
                "content": sample["label"]
            }
        ]
    }

dataset = load_dataset("json", data_files="dataset/dev_task1_release.json")
dataset = dataset['train'].train_test_split(test_size=0.2, shuffle=False)

dataset_train = [format_data(sample) for sample in dataset["train"]]
dataset_test = [format_data(sample) for sample in dataset["test"]]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
)

lora_config = LoraConfig(
    lora_alpha=wandb.config.lora_alpha,
    lora_dropout=wandb.config.lora_dropout,
    target_modules="all-linear",
    r=wandb.config.lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"],
    ensure_weight_tying=True
)

sft_config = SFTConfig(
    output_dir="models",
    max_length=None,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    # gradient_accumulation_steps=8,
    optim="adamw_torch_fused",
    fsdp_config="fsdp2.json",
    logging_steps=5,
    save_strategy="epoch",
    eval_strategy="epoch",
    max_grad_norm=0.3,
    learning_rate=wandb.config.learning_rate,
    bf16=True,
    lr_scheduler_type="constant",
    report_to="wandb",
    dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns = False,
)

def collate_fn(examples):
    texts = []
    images = []
    for example in examples:
        text = processor.apply_chat_template(
            example["messages"], add_generation_prompt=False, tokenize=False
        )
        texts.append(text.strip())
        images.append([example["messages"][1]["content"][1]["image"].convert("RGB")])

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    labels = batch["input_ids"].clone()

    # Mask tokens for not being used in the loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == processor.tokenizer.boi_token_id] = -100
    labels[labels == processor.tokenizer.image_token_id] = -100
    labels[labels == processor.tokenizer.eoi_token_id] = -100

    batch["labels"] = labels
    return batch

trainer = SFTTrainer(
    model=AutoModelForImageTextToText.from_pretrained("models/gemma-4-31B", dtype='auto', quantization_config=bnb_config),
    args=sft_config,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    peft_config=lora_config,
    processing_class=processor,
    data_collator=collate_fn,
)

# trainer.model.print_trainable_parameters()
# if getattr(trainer.accelerator.state, "fsdp_plugin", None):
#    from peft.utils.other import fsdp_auto_wrap_policy

#    fsdp_plugin = trainer.accelerator.state.fsdp_plugin
#    fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

trainer.train()
trainer.save_model("gemma-4-31B-naive")
