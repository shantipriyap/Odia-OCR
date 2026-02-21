#!/usr/bin/env python3
"""
Push the fine-tuned Qwen OCR model to Hugging Face Hub.
Usage:
    export HF_TOKEN="your_token_here"
    python3 push_to_hf.py
"""

import os
import glob
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# Get HF token from environment
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Please set HF_TOKEN environment variable!")

MODEL_PATH = "./qwen_ocr_finetuned"
BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
HF_REPO_ID = "shantipriya/qwen2.5-odia-ocr"

print(f"Pushing model from {MODEL_PATH} to {HF_REPO_ID}...")

# Load processor from base model
processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

# Load base model
print("Loading base model...")
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype="auto",
    trust_remote_code=True,
    device_map="cpu"
)

# Look for the latest checkpoint
import glob
checkpoint_dirs = sorted(glob.glob(f"{MODEL_PATH}/checkpoint-*"), key=lambda x: int(x.split('-')[-1]), reverse=True)

if checkpoint_dirs:
    latest_checkpoint = checkpoint_dirs[0]
    print(f"Loading PEFT adapters from {latest_checkpoint}...")
    model = PeftModel.from_pretrained(base_model, latest_checkpoint)
    # Merge adapters into base model for upload
    print("Merging PEFT adapters into base model...")
    model = model.merge_and_unload()
else:
    print(f"No PEFT checkpoints found at {MODEL_PATH}, using base model directly.")
    model = base_model

# Push to Hub
print("Pushing model to Hub...")
model.push_to_hub(HF_REPO_ID, token=HF_TOKEN)

print("Pushing processor to Hub...")
processor.push_to_hub(HF_REPO_ID, token=HF_TOKEN)

print(f"✓ Upload complete! Model available at: https://huggingface.co/{HF_REPO_ID}")
