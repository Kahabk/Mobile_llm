

# Mobile LLM: 4-bit Optimized Fine-Tuning

This repository contains scripts for fine-tuning and optimizing Large Language Models (LLMs) specifically for **edge devices**. By utilizing **4-bit quantization** and **LoRA (Low-Rank Adaptation)**, these models are designed to run efficiently on hardware with limited VRAM and processing power.

##  Features

* **Memory Efficient:** Fine-tuning using LoRA to reduce hardware requirements.
* **4-bit Quantization:** Optimized for edge deployment (mobile, embedded systems).
* **Dataset Preparation:** Scripts to convert and clean datasets (including DeepMind Math).
* **Seamless Merging:** Easily merge LoRA weights back into the base model.
* **Inference Ready:** Optimized `inference.py` script for testing your 4-bit models.

---

##  Project Structure

| File | Description |
| --- | --- |
| `model_train.py` | Main script for fine-tuning the model using LoRA. |
| `prepare_datasets.py` | Pre-processes raw data into the required training format. |
| `merge_lora.py` | Merges the trained LoRA adapters with the base model weights. |
| `inferance.py` | Runs the model for testing and validation. |
| `convert_deepmind_math.py` | Specific utility for the DeepMind Math dataset. |

---

##  Getting Started

### 1. Prerequisites

Ensure you have the following installed:

* Python 3.10+
* Linux Environment
* NVIDIA GPU (for training) or high-RAM CPU for inference.

### 2. Fine-Tuning

To start the training process with 4-bit optimization:

```bash
python model_train.py

```

### 3. Merging Weights

Once training is complete, merge your adapters:

```bash
python merge_lora.py

```

### 4. Edge Deployment (4-bit)

To run inference on your optimized model:

```bash
python inferance.py

```

---

##  Optimization Details

To achieve high performance on edge devices, this project focuses on:

1. **BitsAndBytes 4-bit Loading:** Reduces model size by up to 75%.
2. **PEFT/LoRA:** Only trains a small fraction of parameters, making it faster and lighter.
3. **Edge Compatibility:** Formatted for easy conversion to `GGUF` or `MLC` formats for mobile deployment.

---
