#!/usr/bin/env python3
"""
Enhanced training script that combines multiple Odia OCR datasets
"""

from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
import torch
import os
import json

# ============================================================================
# MULTI-DATASET LOADING
# ============================================================================

def load_multi_odia_datasets(include_handwritten=True):
    """
    Load and combine multiple Odia OCR datasets
    
    Args:
        include_handwritten: Include tell2jyoti handwritten dataset
    
    Returns:
        Concatenated train and eval datasets
    """
    
    datasets_to_load = [
        ("OdiaGenAIOCR/Odia-lipi-ocr-data", "Current OCR dataset")
    ]
    
    if include_handwritten:
        datasets_to_load.append(
            ("tell2jyoti/odia-handwritten-ocr", "Handwritten OCR dataset")
        )
    
    loaded_datasets = []
    dataset_stats = {}
    
    print("="*70)
    print("📥 LOADING MULTI-ODIA OCR DATASETS")
    print("="*70)
    
    for dataset_id, description in datasets_to_load:
        try:
            print(f"\n🔄 Loading: {dataset_id}")
            print(f"   Description: {description}")
            
            ds = load_dataset(dataset_id)
            
            # Get train split
            if "train" in ds:
                train_split = ds["train"]
                loaded_datasets.append(train_split)
                dataset_stats[dataset_id] = {
                    "samples": len(train_split),
                    "features": list(train_split.features.keys()),
                    "status": "✅"
                }
                print(f"   ✅ Loaded: {len(train_split)} samples")
                print(f"   Features: {list(train_split.features.keys())}")
            else:
                # Try first available split
                first_split = list(ds.keys())[0]
                train_split = ds[first_split]
                loaded_datasets.append(train_split)
                dataset_stats[dataset_id] = {
                    "samples": len(train_split),
                    "features": list(train_split.features.keys()),
                    "split": first_split,
                    "status": "✅"
                }
                print(f"   ✅ Loaded: {len(train_split)} samples (split: {first_split})")
                print(f"   Features: {list(train_split.features.keys())}")
        
        except Exception as e:
            print(f"   ⚠️  Warning: Could not load {dataset_id}")
            print(f"      Error: {str(e)[:100]}")
            dataset_stats[dataset_id] = {
                "status": "❌",
                "error": str(e)[:100]
            }
    
    # Combine datasets
    if loaded_datasets:
        print("\n" + "-"*70)
        total_samples = sum(len(ds) for ds in loaded_datasets)
        combined = concatenate_datasets(loaded_datasets)
        print(f"✅ COMBINED: {total_samples} total samples")
        print(f"   Dataset distribution:")
        for dataset_id, stats in dataset_stats.items():
            if stats["status"] == "✅":
                print(f"   • {dataset_id}: {stats['samples']} samples")
        print("-"*70 + "\n")
        
        return combined, combined, dataset_stats  # Reuse as both train and eval for now
    else:
        print("\n❌ No datasets loaded successfully!")
        raise ValueError("Failed to load any datasets")

# ============================================================================
# MODEL SETUP
# ============================================================================

model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

def find_image_token(processor):
    tokenizer = processor.tokenizer
    candidates = ["<image>", "<Image>", "<img>", "<image_patch>", "<image_token>"]
    for c in candidates:
        try:
            tid = tokenizer.convert_tokens_to_ids(c)
            if tid is not None and hasattr(tokenizer, "unk_token_id") and tid != tokenizer.unk_token_id:
                return c
        except Exception:
            pass
    for t in getattr(tokenizer, "all_special_tokens", []):
        if "image" in t.lower() or "img" in t.lower():
            return t
    return "<image>"

image_token = find_image_token(processor)
if image_token not in processor.tokenizer.get_vocab():
    processor.tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})

# ============================================================================
# LoRA CONFIG - Enhanced for better convergence
# ============================================================================

lora_config = LoraConfig(
    r=32,                              # Increased from 16
    lora_alpha=64,                     # Increased from 32
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,                 # Reduced from 0.1
    bias="none",
)

# ============================================================================
# DATASET PREPROCESSING
# ============================================================================

def preprocess_function(example):
    """Preprocess dataset examples"""
    try:
        image = example["image"].convert("RGB")
    except:
        image = example["image"]
    
    return {
        "image": image,
        "text": example["text"] if "text" in example else example.get("label", "")
    }

# Load combined datasets
train_dataset, eval_dataset, dataset_stats = load_multi_odia_datasets(include_handwritten=True)

# Preprocess
print("🔄 Preprocessing datasets...")
train_dataset = train_dataset.map(preprocess_function, batched=False)
eval_dataset = eval_dataset.map(preprocess_function, batched=False)
print(f"✅ Preprocessed: {len(train_dataset)} training samples\n")

# ============================================================================
# DATA COLLATOR
# ============================================================================

class QwenOCRDataCollator:
    def __init__(self, processor):
        self.processor = processor
        try:
            self.image_token = find_image_token(processor)
        except Exception:
            self.image_token = "<image>"
        self.num_image_tokens = 1
    
    def __call__(self, batch):
        images = [example["image"] for example in batch]
        texts = [example["text"] for example in batch]
        texts = [f"{self.image_token} {t}" for t in texts]

        inputs = self.processor(
            images,
            text=texts,
            padding=True,
            truncation=False,
            return_tensors="pt"
        )
        
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

data_collator = QwenOCRDataCollator(processor)

# ============================================================================
# IMPROVED TRAINING ARGUMENTS
# ============================================================================

training_args = TrainingArguments(
    output_dir="./qwen_ocr_finetuned_multi",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,                # Improved: lowered from 2e-4
    max_steps=500,                     # Improved: increased from 100
    warmup_steps=50,                   # Improved: added warmup
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    eval_steps=50,                     # Improved: added evaluation
    evaluation_strategy="steps",       # Improved: track metrics
    lr_scheduler_type="cosine",        # Improved: cosine decay
    fp16=False,
    remove_unused_columns=False,
    dataloader_num_workers=0,
    optim="adamw_torch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

print("="*70)
print("📊 TRAINING CONFIGURATION")
print("="*70)
print(f"Model: {model_name}")
print(f"Max Steps: {training_args.max_steps}")
print(f"Warmup Steps: {training_args.warmup_steps}")
print(f"Learning Rate: {training_args.learning_rate}")
print(f"LR Scheduler: {training_args.lr_scheduler_type}")
print(f"LoRA Rank: {lora_config.r}")
print(f"LoRA Alpha: {lora_config.lora_alpha}")
print(f"Train Dataset Size: {len(train_dataset)}")
print(f"Eval Dataset Size: {len(eval_dataset)}")
print("="*70 + "\n")

# ============================================================================
# TRAINING
# ============================================================================

def quick_forward_check(n=2):
    batch = [train_dataset[i] for i in range(min(n, len(train_dataset)))]
    inputs = data_collator(batch)
    if 'model' in globals() and globals()['model'] is not None:
        return globals()['model'](**inputs)
    return inputs

trainer = None

if __name__ == "__main__":
    print("🚀 INITIALIZING MODEL...")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="auto",
    )

    model.resize_token_embeddings(len(processor.tokenizer))
    model = get_peft_model(model, lora_config)
    
    print(f"✅ Model initialized: {model_name}")
    print(f"   LoRA parameters: Trainable\n")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("🚀 STARTING TRAINING...\n")
    trainer.train()
    
    # Save dataset stats
    with open("training_dataset_stats.json", "w") as f:
        json.dump(dataset_stats, f, indent=2)
    
    print("\n✅ Training complete!")
    print(f"   Stats saved to: training_dataset_stats.json")
