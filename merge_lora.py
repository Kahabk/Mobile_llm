import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2-0.5B-Instruct"
LORA_PATH = "./qwen_study_lora"
OUT_PATH = "./qwen_study_merged"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    LORA_PATH,
    trust_remote_code=True
)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",  
    trust_remote_code=True
)

print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(model, LORA_PATH)

print("Merging LoRA into base model...")
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained(OUT_PATH, safe_serialization=True)
tokenizer.save_pretrained(OUT_PATH)

print(" Merge complete. Saved to:", OUT_PATH)
