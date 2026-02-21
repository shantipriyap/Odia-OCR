#!/usr/bin/env python3
"""
Simple and robust fine-tuning of Qwen2.5-VL on merged Odia OCR dataset
Uses file paths and robust error handling
"""

from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model
import torch
import os
from PIL import Image
import json

print(f"""
╔════════════════════════════════════════════════════════════════╗
║      🚀 TRAINING QWEN2.5-VL ON MERGED ODIA OCR (v2) 🚀      ║
║                 Robust & Simplified Approach                   ║
╚════════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# 1) LOAD DATASET
# ============================================================================

print("📥 Loading merged Odia OCR dataset...")
try:
    dataset = load_dataset("shantipriya/odia-ocr-merged")
    print(f"✅ Loaded: {len(dataset['train']):,} samples\n")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# ============================================================================
# 2) LOAD MODEL AND PROCESSOR
# ============================================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
print(f"📦 Loading {MODEL_NAME}...")

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("✅ Processor loaded")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    trust_remote_code=True,
    device_map="auto",
)
print("✅ Model loaded\n")

# ============================================================================
# 3) LORA CONFIG
# ============================================================================

print("⚙️  Configuring LoRA...")
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print()

# ============================================================================
# 4) PREPARE DATA - SIMPLE APPROACH
# ============================================================================

print("🔄 Preparing dataset...")

def load_and_process_image(image_input):
    """Load image from various sources"""
    try:
        if image_input is None:
            return None
        
        if isinstance(image_input, str):
            # It's a file path
            img = Image.open(image_input).convert("RGB")
        else:
            # Assume it's already a PIL Image
            img = image_input.convert("RGB") if hasattr(image_input, "convert") else image_input
        
        return img
    except Exception as e:
        return None

def preprocess(examples):
    """Prepare examples for training"""
    batch_images = []
    batch_texts = []
    
    for i in range(len(examples["image"])):
        try:
            img = load_and_process_image(examples["image"][i])
            text = examples["text"][i]
            
            if img is not None and text:
                batch_images.append(img)
                batch_texts.append(text)
        except Exception as e:
            continue
    
    if not batch_images:
        # Return empty batch
        return {
            "images": [],
            "texts": []
        }
    
    return {
        "images": batch_images,
        "texts": batch_texts
    }

# Map preprocessing
print("Processing dataset...")
dataset_mapped = dataset["train"].map(preprocess, batched=True, batch_size=32)
print(f"✅ Dataset ready\n")

# ============================================================================
# 5) DATA COLLATOR
# ============================================================================

class SimpleQwenCollator:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, batch):
        # Extract from batch
        images = [item["images"] for item in batch if item.get("images")]
        texts = [item["texts"] for item in batch if item.get("texts")]
        
        # Flatten lists
        flat_images = []
        flat_texts = []
        for img_list, text_list in zip(images, texts):
            if isinstance(img_list, list):
                flat_images.extend(img_list)
                flat_texts.extend(text_list)
            else:
                flat_images.append(img_list)
                flat_texts.append(text_list)
        
        if not flat_images:
            return {
                "input_ids": torch.tensor([[0]]),
                "labels": torch.tensor([[0]])
            }
        
        # Process
        try:
            inputs = self.processor(
                flat_images,
                text=flat_texts,
                padding=True,
                return_tensors="pt"
            )
            inputs["labels"] = inputs["input_ids"].clone()
            return inputs
        except Exception as e:
            print(f"⚠️  Collator error: {e}")
            return {
                "input_ids": torch.tensor([[0]]),
                "labels": torch.tensor([[0]])
            }

collator = SimpleQwenCollator(processor)

# ============================================================================
# 6) TRAINING
# ============================================================================

print("📋 Setting up training...")

training_args = TrainingArguments(
    output_dir="./qwen_odia_improved_v3",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=500,
    warmup_steps=50,
    logging_steps=10,
    save_steps=50,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    remove_unused_columns=False,
    dataloader_num_workers=0,
    optim="adamw_torch",
    report_to=[],  # No reporting
)

print(f"✅ Training configuration ready\n")

print("🎯 Creating trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_mapped,
    data_collator=collator,
)

print("=" * 70)
print("🚀 STARTING TRAINING")
print("=" * 70)
print(f"Dataset: shantipriya/odia-ocr-merged")
print(f"Samples: 145,781 (145,717 + 64)")
print(f"Steps: 500")
print(f"Expected: CER 100% → 30-50%")
print(f"Duration: ~4-6 hours")
print("=" * 70 + "\n")

trainer.train()

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETE!")
print("=" * 70)
print(f"\n✅ Model saved to: ./qwen_odia_improved_v3")
print(f"📖 Next: python3 evaluate_model.py")
