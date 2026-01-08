import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
LORA_WEGHTS = "./qwen_study_lora"

tokenizer = AutoTokenizer.from_pretrained(
    LORA_WEGHTS,
    trust_remote_code=True)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model,LORA_WEGHTS)
model.eval()
def infer(question, context=""):
    prompt = f"""
You are a mathematics tutor.

Rules:
- Solve step by step.
- Number each step.
- Do not skip algebra steps.
- Check your result before concluding.

Context:
{context}

Problem:
{question}

Solution:
Step 1:
"""


    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )


    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Return only the generated explanation
    return text.split("Step-by-step solution:")[-1].strip()


if __name__ == "__main__":
    
    print(infer("What is photosynthesis"))
            