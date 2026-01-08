import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

# -----------------------------
# Quantization (QLoRA)
# -----------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)


# -----------------------------
# LoRA config (STABLE)
# -----------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)


model = get_peft_model(model, lora_config)
model.config.use_cache = False
# -----------------------------
# Prompt formatting
# -----------------------------
def format_prompt(example):
    return (
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        "### Context:\n"
        f"{example['context']}\n\n"
        "### Question:\n"
        f"{example['question']}\n\n"
        "### Answer:\n"
        f"{example['answer']}"
    )

# -----------------------------
# Load datasets
# -----------------------------
train_ds = load_dataset(
    "json",
    data_files="train_split.jsonl",
    split="train"
)

val_ds = load_dataset(
    "json",
    data_files="val_split.jsonl",
    split="train"
)

# -----------------------------
# Training arguments (PRO)
# -----------------------------
training_args = TrainingArguments(
    output_dir="./qwen_study_lora",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=1,

    fp16=False,
    bf16=False,
    max_grad_norm=0.0,

    eval_strategy="steps",
    eval_steps=5000,
    save_steps=5000,

    logging_steps=100,
    save_total_limit=2,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    optim="paged_adamw_8bit",
    report_to="none"
)






# -----------------------------
# Trainer (CORRECT API)
# -----------------------------
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    formatting_func=format_prompt,
    processing_class=tokenizer
)

trainer.train()

trainer.save_model("./qwen_study_lora")
tokenizer.save_pretrained("./qwen_study_lora")
