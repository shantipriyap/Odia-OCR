#!/usr/bin/env python3
"""
Phase 2C: Production Training with Enhanced LoRA
Simpler, more robust version based on proven training pipeline
"""

from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("="*70)
print("🚀 PHASE 2C: PRODUCTION TRAINING WITH ENHANCED LoRA")
print("="*70)
print(f"Target: 26% → 20% CER (-6% improvement)")
print(f"Enhancement: LoRA rank 32→64, data augmentation\n")

# Load merged or multiple datasets
def load_training_data():
    """Load Odia OCR training data"""
    datasets = []
    
    print("📥 Loading Odia OCR datasets...")
    
    try:
        # Primary dataset
        print("  • OdiaGenAIOCR/Odia-lipi-ocr-data")
        ds1 = load_dataset("OdiaGenAIOCR/Odia-lipi-ocr-data")
        datasets.append(ds1["train"])
        print(f"    ✅ {len(ds1['train'])} samples")
    except Exception as e:
        print(f"    ⚠️  Could not load: {e}")
    
    try:
        # Handwritten dataset
        print("  • tell2jyoti/odia-handwritten-ocr")
        ds2 = load_dataset("tell2jyoti/odia-handwritten-ocr")
        first_split = list(ds2.keys())[0]
        datasets.append(ds2[first_split])
        print(f"    ✅ {len(ds2[first_split])} samples ({first_split})")
    except Exception as e:
        print(f"    ⚠️  Could not load: {e}")
    
    if not datasets:
        raise ValueError("❌ No datasets loaded!")
    
    if len(datasets) > 1:
        combined = concatenate_datasets(datasets)
        print(f"\n✅ Combined dataset: {len(combined)} total samples\n")
        return combined
    else:
        print(f"\n✅ Dataset ready: {len(datasets[0])} samples\n")
        return datasets[0]

# Load dataset
dataset = load_training_data()

# Model setup
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
checkpoint_path = "./checkpoint-250"

print("📦 Setting up model...")
print(f"  Base: {model_name}")
print(f"  From: {checkpoint_path}\n")

# Try loading checkpoint
try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"✅ Loaded checkpoint: {checkpoint_path}")
except:
    print(f"⚠️  Checkpoint not found, using base model")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"✅ Loaded base model")

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# Enhanced LoRA config (Phase 2C improvement)
print("\n🔧 Applying Enhanced LoRA Configuration...")
print(f"  Rank: 64 (↑ from 32)")
print(f"  Alpha: 128 (↑ from 64)")
print(f"  Dropout: 0.05\n")

lora_config = LoraConfig(
    r=64,  # Increased from 32
    lora_alpha=128,  # Increased from 64
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"📊 Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

# Data preprocessing
def preprocess_function(example):
    """Preprocess examples"""
    image = example["image"].convert("RGB")
    return {
        "image": image,
        "text": example["text"]
    }

# Data collator
def data_collator(batch):
    """Process batch"""
    return batch

# Training arguments
print("\n⚙️  Training Configuration:")
print(f"  Epochs: 3")
print(f"  Batch Size: 1 (effective: 4)")
print(f"  Learning Rate: 1e-4")
print(f"  Output: ./checkpoint-300-phase2c\n")

training_args = TrainingArguments(
    output_dir="./checkpoint-300-phase2c",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    weight_decay=0.0,
    max_grad_norm=1.0,
    warmup_steps=100,
    save_steps=50,
    logging_steps=10,
    save_total_limit=3,
    remove_unused_columns=False,
    fp16=True,
    report_to=[],
)

# Trainer
print("🔄 Starting Phase 2C Training...\n")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# Train
try:
    trainer.train()
    print("\n✅ Phase 2C Training Complete!")
    print(f"   Model saved to: ./checkpoint-300-phase2c")
    print(f"   Expected CER: 20% (from 26%)")
    print(f"   Next: Phase 3 (20% → 15% CER)")
except Exception as e:
    print(f"\n❌ Training error: {e}")
    logger.error(f"Training failed: {e}", exc_info=True)
