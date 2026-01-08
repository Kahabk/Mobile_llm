import json
import os
from datasets import load_dataset

# ------------------------
# Setup
# ------------------------
os.makedirs("dataset", exist_ok=True)

# ------------------------
# GSM8K (Socratic)
# ------------------------
print("Loading GSM8K (socratic)...")
gsm8k = load_dataset("openai/gsm8k", "socratic")

with open("dataset/gsm8k.jsonl", "w") as f:
    for ex in gsm8k["train"]:
        entry = {
            "instruction": "Solve the math problem step by step.",
            "context": "",
            "question": ex["question"],
            "answer": ex["answer"]
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print("Saved dataset/gsm8k.jsonl")

# ------------------------
# MathQA (optional)
# ------------------------
print("\nPreparing MathQA (optional)...")

if not os.path.exists("mathqa_raw"):
    print("WARNING: mathqa_raw/ not found. Skipping MathQA.")
else:
    mathqa_files = [f for f in os.listdir("mathqa_raw") if f.endswith(".json")]

    if not mathqa_files:
        print("WARNING: mathqa_raw/ exists but has no JSON files. Skipping MathQA.")
    else:
        with open("dataset/mathqa.jsonl", "w") as f:
            for file in mathqa_files:
                with open(os.path.join("mathqa_raw", file)) as jf:
                    data = json.load(jf)

                for ex in data:
                    entry = {
                        "instruction": "Solve the math problem step by step.",
                        "context": "",
                        "question": ex.get("question_text", ""),
                        "answer": ex.get("answer_text", "")
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print("Saved dataset/mathqa.jsonl")

# ------------------------
# SQuAD 2.0
# ------------------------
print("\nLoading SQuAD v2...")
squad = load_dataset("squad_v2")

with open("dataset/squad2.jsonl", "w") as f:
    for ex in squad["train"]:
        answer_text = ""
        if ex["answers"]["text"]:
            answer_text = ex["answers"]["text"][0]

        entry = {
            "instruction": "Answer the question using the given context.",
            "context": ex["context"],
            "question": ex["question"],
            "answer": answer_text
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print("Saved dataset/squad2.jsonl")
print("\nDataset preparation complete.")
