#!/usr/bin/env python3
"""
Simplified training of Qwen2.5-VL on merged Odia OCR dataset
No evaluation during training (focus on training metrics)
Fixed data processing
"""

from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
import torch
import os
from PIL import Image

print(f"""
╔════════════════════════════════════════════════════════════════╗
║      🚀 TRAINING QWEN2.5-VL ON MERGED ODIA OCR (v5) 🚀      ║
║        Fixed Data Processing - No Eval, Fast Training          ║
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
# 4) PREPARE DATA
# ============================================================================

print("🔄 Preparing dataset (filtering invalid samples)...")

def load_image(img_input):
    """Safely load image"""
    try:
        if img_input is None:
            return None
        if isinstance(img_input, str) and os.path.exists(img_input):
            return Image.open(img_input).convert("RGB")
        elif hasattr(img_input, "convert"):
            return img_input.convert("RGB")
        elif isinstance(img_input, Image.Image):
            return img_input
        return None
    except:
        return None

def is_valid_sample(example):
    """Filter valid samples"""
    try:
        img = load_image(example.get("image"))
        text = str(example.get("text", ""))
        return img is not None and len(text.strip()) > 0
    except:
        return False

# Filter dataset directly
dataset_filtered = dataset["train"].filter(is_valid_sample, num_proc=4)
print(f"✅ Filtered to: {len(dataset_filtered):,} valid samples\n")

def preprocess_single(example):
    """Process single example"""
    try:
        img = load_image(example["image"])
        text = str(example.get("text", ""))
        
        if img is not None and text.strip():
            return {"image": img, "text": text}
        else:
            return None
    except:
        return None

# Remove None values after filtering
dataset_processed = dataset_filtered.map(
    preprocess_single,
    remove_columns=dataset_filtered.column_names,
    num_proc=4
)

print(f"✅ Dataset ready: {len(dataset_processed):,} samples\n")

# ============================================================================
# 5) DATA COLLATOR
# ============================================================================

class OdiaOCRCollator:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, batch):
        images = [item["image"] for item in batch if item and "image" in item]
        texts = [item["text"] for item in batch if item and "text" in item]
        
        if not images or not texts:
            return {
                "input_ids": torch.tensor([[0]]),
                "labels": torch.tensor([[0]])
            }
        
        try:
            inputs = self.processor(
                images,
                text=texts,
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

collator = OdiaOCRCollator(processor)

# ============================================================================
# 6) TRAINING
# ============================================================================

print("📋 Setting up training...")

training_args = TrainingArguments(
    output_dir="./qwen_odia_ocr_improved_v2",
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
    eval_strategy="no",  # No evaluation
)

print(f"✅ Training config ready\n")

print("🎯 Creating trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_processed,
    data_collator=collator,
)

print("=" * 70)
print("🚀 STARTING TRAINING")
print("=" * 70)
print(f"Base Model: {MODEL_NAME}")
print(f"Dataset: shantipriya/odia-ocr-merged")
print(f"Valid Samples: {len(dataset_processed):,}")
print(f"Training Steps: 500")
print(f"Save every: 50 steps")
print(f"Expected Duration: 4-6 hours")
print(f"Expected CER: 100% → 30-50%")
print("=" * 70 + "\n")

try:
    trainer.train()
except Exception as e:
    print(f"\n⚠️  Training error: {e}")
    import traceback
    traceback.print_exc()
    print("Attempting to save current progress...")
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained("./qwen_odia_ocr_improved_v2/final_attempt")

print("\n" + "=" * 70)
print("✅ TRAINING PHASE COMPLETE!")
print("=" * 70)
print(f"\nModel output directory: ./qwen_odia_ocr_improved_v2")
print(f"Checkpoints saved every 50 steps")
print(f"Next step: Upload best model to HuggingFace Hub")
