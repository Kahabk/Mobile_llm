import json
import glob
import os

OUT = "dataset/deepmind_math.jsonl"
os.makedirs("dataset", exist_ok=True)

use_dirs = [
    "train-easy",
    "train-medium"
]

base = os.path.expanduser("~/mobil_llm/math_data")

count = 0
with open(OUT, "w") as out:
    for d in use_dirs:
        for file in glob.glob(f"{base}/{d}/*.txt"):
            with open(file) as f:
                lines = f.readlines()
                for i in range(0, len(lines) - 1, 2):
                    q = lines[i].strip()
                    a = lines[i + 1].strip()
                    sample = {
                        "instruction": "Solve the mathematics problem step-by-step and explain clearly for a student.",
                        "context": "",
                        "question": q,
                        "answer": a
                    }
                    out.write(json.dumps(sample) + "\n")
                    count += 1

print(f"Saved {count} math samples to {OUT}")
