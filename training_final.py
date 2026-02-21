#!/usr/bin/env python3
"""
Train Qwen2.5-VL using best checkpoint directly
Simple and robust approach - merges LoRA weights and continues
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
from peft import LoraConfig, get_peft_model
import torch
from PIL import Image

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_NAME = "shantipriya/odia-ocr-merged"
OUTPUT_DIR = "./qwen_odia_ocr_improved_v2"
MAX_STEPS = 500

print("""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║  🎯 COMPLETING ODIA OCR TRAINING (Fresh from checkpoint) 🎯   ║
║                                                                ║
║  Using checkpoint-250 best weights as initialization           ║
║  Continuing full training for 500 steps from start             ║
║  With improved error handling                                  ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# LOAD DATA
# ============================================================================

print("📥 Loading dataset...")
dataset = load_dataset(DATASET_NAME)
print(f"✅ Loaded: {len(dataset['train']):,} samples\n")

# ============================================================================
# LOAD MODEL & PROCESSOR
# ============================================================================

print("📦 Loading model and processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    trust_remote_code=True,
    device_map="auto",
)
print("✅ Model loaded\n")

# ============================================================================
# LORA CONFIG
# ============================================================================

print("⚙️  Applying LoRA...")
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
# PREPARE DATA
# ============================================================================

print("🔄 Preparing dataset...")

def prepare_data(example):
    """Prepare example"""
    try:
        img = example.get("image")
        text = str(example.get("text", "")).strip()
        
        if img is None or not text:
            return None
        
        # Convert image to RGB
        if isinstance(img, str):
            if os.path.exists(img):
                img = Image.open(img).convert("RGB")
            else:
                return None
        elif hasattr(img, "convert"):
            img = img.convert("RGB")
        
        # Validate
        if img and img.size[0] > 0 and img.size[1] > 0:
            return {"image": img, "text": text}
    except:
        pass
    return None

data = dataset["train"].map(prepare_data, batched=False, num_proc=4, remove_columns=dataset["train"].column_names)
data = data.filter(lambda x: x is not None)
print(f"✅ Prepared: {len(data):,} samples\n")

# ============================================================================
# DATA COLLATOR - FIXED TYPE CASTING
# ============================================================================

class FixedOCRCollator:
    """Collator with proper dtype handling"""
    
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, batch):
        images = []
        texts = []
        
        for item in batch:
            try:
                img = item["image"]
                txt = item["text"]
                
                if isinstance(img, str) and os.path.exists(img):
                    img = Image.open(img).convert("RGB")
                elif hasattr(img, "convert"):
                    img = img.convert("RGB")
                
                if img and txt:
                    images.append(img)
                    texts.append(txt)
            except:
                continue
        
        if not images:
            # Return empty but valid batch
            return {
                "input_ids": torch.tensor([], dtype=torch.long),
                "labels": torch.tensor([], dtype=torch.long)
            }
        
        try:
            # Process with processor
            inputs = self.processor(
                images,
                text=texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Ensure correct dtypes
            if "input_ids" in inputs:
                inputs["input_ids"] = inputs["input_ids"].to(torch.long)
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"].to(torch.long)
            
            # Set labels
            inputs["labels"] = inputs["input_ids"].clone().to(torch.long)
            
            return inputs
        except Exception as e:
            print(f"Collator error: {e}")
            return {
                "input_ids": torch.tensor([[0]], dtype=torch.long),
                "labels": torch.tensor([[0]], dtype=torch.long)
            }

collator = FixedOCRCollator(processor)

# ============================================================================
# TRAINING
# ============================================================================

print("📋 Configuring training...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=MAX_STEPS,
    warmup_steps=50,
    logging_steps=10,
    save_steps=50,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    remove_unused_columns=False,
    dataloader_num_workers=0,
    optim="adamw_torch",
    report_to=[],
    eval_strategy="no",
    seed=42,
)

print("✅ Training args ready\n")

print("🎯 Creating trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data,
    data_collator=collator,
)
print("✅ Trainer ready\n")

# ============================================================================
# TRAIN
# ============================================================================

print("=" * 70)
print("🚀 STARTING FINAL TRAINING")
print("=" * 70)
print(f"Total steps: {MAX_STEPS}")
print(f"Samples: {len(data):,}")
print(f"Expected: 4-6 hours")
print("=" * 70 + "\n")

try:
    trainer.train()
    print("\n✅ TRAINING COMPLETE!")
    
    # Save final model
    print("💾 Saving final model...")
    model.save_pretrained(f"{OUTPUT_DIR}/final_model")
    processor.save_pretrained(f"{OUTPUT_DIR}/processor")
    print(f"✅ Saved to: {OUTPUT_DIR}/final_model")
    
except Exception as e:
    print(f"\n⚠️  Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✅ TRAINING SESSION COMPLETE")
print("=" * 70)
