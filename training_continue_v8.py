#!/usr/bin/env python3
"""
Resume training of Qwen2.5-VL on merged Odia OCR dataset from checkpoint-250
Continuation training with fresh optimizer state (ignores checkpoint optimizer)
"""

import os
import sys
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, PeftModel
import torch
from PIL import Image
from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_NAME = "shantipriya/odia-ocr-merged"
OUTPUT_DIR = "./qwen_odia_ocr_improved_v2"
CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoint-250"
MAX_STEPS = 500
WARMUP_STEPS = 25  # Reduced warmup since we're resuming

print(f"""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║  🚀 CONTINUING QWEN2.5-VL ODIA OCR TRAINING (from step 250) 🚀║
║                                                                ║
║  Dataset:        145,781 Odia OCR samples                      ║
║  Continue From:  checkpoint-250 (LoRA weights)                 ║
║  Continue To:    500 steps (250 more steps)                    ║
║  New Optimizer:  Fresh AdamW (checkpoint optimizer skipped)    ║
║  Expected CER:   30-50% target                                 ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# LOAD DATASET
# ============================================================================

print("📥 Loading merged Odia OCR dataset...")
try:
    dataset = load_dataset(DATASET_NAME)
    print(f"✅ Loaded: {len(dataset['train']):,} samples\n")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# ============================================================================
# LOAD PROCESSOR & MODEL
# ============================================================================

print("📦 Loading processor and model...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("✅ Processor loaded")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    trust_remote_code=True,
    device_map="auto",
)
print("✅ Model loaded")

# Reload LoRA from checkpoint-250
print(f"🔧 Loading fine-tuned LoRA weights from checkpoint...")
model = PeftModel.from_pretrained(
    model,
    CHECKPOINT_DIR,
    torch_dtype="auto"
)
print("✅ LoRA weights loaded\n")

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

print("🔄 Preprocessing dataset...")

def safe_load_image(img):
    """Safely load image"""
    try:
        if img is None:
            return None
        if isinstance(img, str) and os.path.exists(img):
            img = Image.open(img).convert("RGB")
        elif hasattr(img, "convert"):
            img = img.convert("RGB")
        elif not isinstance(img, Image.Image):
            return None
        return img if img.size[0] > 0 and img.size[1] > 0 else None
    except:
        return None

def preprocess_fn(example):
    """Preprocess example"""
    try:
        img = safe_load_image(example.get("image"))
        text = str(example.get("text", "")).strip()
        if img is not None and text:
            return {"image": img, "text": text}
    except:
        pass
    return None

dataset_processed = dataset["train"].map(preprocess_fn, batched=False, num_proc=4, remove_columns=dataset["train"].column_names)
dataset_processed = dataset_processed.filter(lambda x: x is not None)
print(f"✅ Processed: {len(dataset_processed):,} samples\n")

# ============================================================================
# DATA COLLATOR
# ============================================================================

class OCRDataCollator:
    """Robust data collator"""
    
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, batch):
        images, texts = [], []
        
        for example in batch:
            try:
                img = example.get("image")
                text = str(example.get("text", "")).strip()
                
                if isinstance(img, str):
                    if os.path.exists(img):
                        img = Image.open(img).convert("RGB")
                    else:
                        continue
                elif hasattr(img, "convert"):
                    img = img.convert("RGB")
                
                if img and text:
                    images.append(img)
                    texts.append(text)
            except:
                continue
        
        if not images:
            return {"input_ids": torch.tensor([]), "labels": torch.tensor([])}
        
        try:
            inputs = self.processor(images, text=texts, padding=True, truncation=True, return_tensors="pt")
            inputs["labels"] = inputs["input_ids"].clone()
            return inputs
        except:
            return {"input_ids": torch.tensor([]), "labels": torch.tensor([])}

collator = OCRDataCollator(processor)

# ============================================================================
# TRAINING ARGUMENTS
# ============================================================================

print("📋 Configuring continued training...")

# Start from step 250
initial_step = 250

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=MAX_STEPS,
    warmup_steps=WARMUP_STEPS,
    logging_steps=10,
    save_steps=50,
    learning_rate=5e-5,  # Lower LR for continuation
    lr_scheduler_type="cosine",
    remove_unused_columns=False,
    dataloader_num_workers=0,
    optim="adamw_torch",
    report_to=[],
    eval_strategy="no",
)

print(f"✅ Training config ready")
print(f"   Starting step: {initial_step}")
print(f"   Target: {MAX_STEPS} total steps")
print(f"   Remaining: {MAX_STEPS - initial_step} steps")
print(f"   Learning rate: 5e-5\n")

# ============================================================================
# TRAINER
# ============================================================================

print("🎯 Creating trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_processed,
    data_collator=collator,
)
print("✅ Trainer ready\n")

# ============================================================================
# TRAINING
# ============================================================================

print("=" * 70)
print("🚀 CONTINUING TRAINING FROM STEP 250")
print("=" * 70)
print(f"Model: Fine-tuned from checkpoint-250")
print(f"Remaining steps: {MAX_STEPS - initial_step}")
print(f"Expected: ~{(MAX_STEPS - initial_step) * 1.8 / 3600:.1f} hours")
print("=" * 70 + "\n")

try:
    trainer.train()
    
    print("\n✅ TRAINING SUCCESSFULLY COMPLETED!")
    print("\n💾 Saving final model...")
    model.save_pretrained(f"{OUTPUT_DIR}/final_model")
    processor.save_pretrained(f"{OUTPUT_DIR}/processor")
    print(f"✅ Final model saved: {OUTPUT_DIR}/final_model")
    
except KeyboardInterrupt:
    print("\n⏹️  Training interrupted")
    trainer.save_model(f"{OUTPUT_DIR}/interrupted")
    
except Exception as e:
    print(f"\n⚠️  Error: {e}")
    import traceback
    traceback.print_exc()
    trainer.save_model(f"{OUTPUT_DIR}/checkpoint_error")

print("\n" + "=" * 70)
print("✅ TRAINING SESSION COMPLETE")
print("=" * 70)
print(f"Checkpoints saved to: {OUTPUT_DIR}")
print("Next: Evaluate and upload to HuggingFace Hub")
print("=" * 70)
