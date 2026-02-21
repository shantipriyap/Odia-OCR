#!/usr/bin/env python3
"""
Enhanced training script combining multiple large-scale Odia OCR datasets
- OdiaGenAIOCR: 64 samples
- tell2jyoti/odia-handwritten-ocr: 182,152 character samples
- darknight054/indic-mozhi-ocr: 1.2M+ Indic words (includes Odia)
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
# MULTI-DATASET LOADING - COMPREHENSIVE
# ============================================================================

def load_comprehensive_odia_datasets(
    use_odiagenaiocr=True,
    use_tell2jyoti=True,
    use_indic_mozhi=False,  # Large dataset - optional
    max_samples_per_dataset=None
):
    """
    Load and combine multiple Odia OCR datasets
    
    Args:
        use_odiagenaiocr: Include OdiaGenAIOCR base dataset
        use_tell2jyoti: Include tell2jyoti handwritten characters (182K samples)
        use_indic_mozhi: Include darknight054/indic-mozhi-ocr (1.2M+ samples, requires filtering for Odia)
        max_samples_per_dataset: Limit samples per dataset (for testing)
    
    Returns:
        Combined train and eval datasets, plus statistics
    """
    
    print("="*80)
    print("📥 LOADING COMPREHENSIVE ODIA OCR DATASETS")
    print("="*80 + "\n")
    
    loaded_datasets = []
    dataset_stats = {}
    
    # 1) OdiaGenAIOCR Base Dataset
    if use_odiagenaiocr:
        try:
            print("🔄 [1/3] Loading OdiaGenAIOCR/Odia-lipi-ocr-data...")
            ds = load_dataset("OdiaGenAIOCR/Odia-lipi-ocr-data")
            train_split = ds["train"]
            num_samples = len(train_split)
            loaded_datasets.append(train_split)
            dataset_stats["OdiaGenAIOCR/Odia-lipi-ocr-data"] = {
                "samples": num_samples,
                "features": list(train_split.features.keys()),
                "type": "word-level images",
                "status": "✅"
            }
            print(f"   ✅ Loaded: {num_samples} samples (word-level)")
            print(f"   Features: {list(train_split.features.keys())}")
        except Exception as e:
            print(f"   ⚠️  Error: {e}")
    
    # 2) tell2jyoti Handwritten Character Dataset
    if use_tell2jyoti:
        try:
            print("\n🔄 [2/3] Loading tell2jyoti/odia-handwritten-ocr...")
            ds = load_dataset("tell2jyoti/odia-handwritten-ocr")
            train_split = ds["train"]
            num_samples = len(train_split)
            loaded_datasets.append(train_split)
            dataset_stats["tell2jyoti/odia-handwritten-ocr"] = {
                "samples": num_samples,
                "features": list(train_split.features.keys()),
                "type": "character-level (32x32 images)",
                "status": "✅"
            }
            print(f"   ✅ Loaded: {num_samples} samples (character-level)")
            print(f"   Features: {list(train_split.features.keys())}")
            print(f"   Note: 47 OHCS characters, 182K+ handwritten samples")
        except Exception as e:
            print(f"   ⚠️  Error: {e}")
    
    # 3) darknight054 Indic Mozhi (Odia words)
    if use_indic_mozhi:
        try:
            print("\n🔄 [3/3] Loading darknight054/indic-mozhi-ocr (Odia split)...")
            ds = load_dataset("darknight054/indic-mozhi-ocr", "oriya")
            train_split = ds["train"]
            num_samples = len(train_split)
            
            # Limit if requested (for testing)
            if max_samples_per_dataset:
                train_split = train_split.select(range(min(max_samples_per_dataset, num_samples)))
                num_samples = len(train_split)
            
            loaded_datasets.append(train_split)
            dataset_stats["darknight054/indic-mozhi-ocr (Odia)"] = {
                "samples": num_samples,
                "features": list(train_split.features.keys()),
                "type": "printed word-level images (1.2M Indic total)",
                "status": "✅"
            }
            print(f"   ✅ Loaded: {num_samples:,} samples (printed Odia words)")
            print(f"   Features: {list(train_split.features.keys())}")
        except Exception as e:
            print(f"   ⚠️  Error: {e}")
            print(f"      Note: Odia subset may not be available. Using general approach.")
    
    # Combine all datasets
    print("\n" + "-"*80)
    if loaded_datasets:
        combined = concatenate_datasets(loaded_datasets)
        total_samples = len(combined)
        print(f"✅ COMBINED DATASET: {total_samples:,} total samples")
        print()
        for dataset_name, stats in dataset_stats.items():
            print(f"   • {dataset_name}")
            print(f"     - Samples: {stats['samples']:,}")
            print(f"     - Type: {stats['type']}")
        print("-"*80 + "\n")
        
        return combined, combined, dataset_stats  # Reuse as both train and eval
    else:
        raise ValueError("❌ No datasets loaded successfully!")

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
# LoRA CONFIG - Optimized for large-scale training
# ============================================================================

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

# ============================================================================
# DATASET PREPROCESSING
# ============================================================================

def preprocess_function(example):
    """Preprocess mixed dataset formats"""
    try:
        # Handle image field - multiple possible names
        if "image" in example:
            image = example["image"]
        elif "image_path" in example:
            from PIL import Image
            image = Image.open(example["image_path"]).convert("RGB")
        else:
            raise KeyError("No image field found")
        
        # Ensure RGB
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        
        # Handle text field - multiple possible names
        if "text" in example:
            text = example["text"]
        elif "character" in example:
            text = example["character"]
        elif "label" in example:
            text = example["label"]
        else:
            text = ""
        
        return {
            "image": image,
            "text": str(text) if text else ""
        }
    except Exception as e:
        print(f"Error preprocessing example: {e}")
        return None

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
        # Filter out None values
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        
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

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n🚀 ODIA OCR COMPREHENSIVE MULTI-DATASET TRAINING")
    print("="*80 + "\n")
    
    # Load datasets
    train_dataset, eval_dataset, dataset_stats = load_comprehensive_odia_datasets(
        use_odiagenaiocr=True,
        use_tell2jyoti=True,
        use_indic_mozhi=False,  # Set to True after testing basic setup
        max_samples_per_dataset=None
    )
    
    # Preprocess
    print("🔄 Preprocessing datasets...")
    train_dataset = train_dataset.map(preprocess_function, batched=False, remove_columns=True)
    eval_dataset = eval_dataset.map(preprocess_function, batched=False, remove_columns=True)
    print(f"✅ Preprocessed: {len(train_dataset):,} training samples\n")
    
    data_collator = QwenOCRDataCollator(processor)
    
    # ========================================================================
    # TRAINING ARGUMENTS - Optimized for large-scale data
    # ========================================================================
    
    training_args = TrainingArguments(
        output_dir="./qwen_ocr_finetuned_comprehensive",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=1000,                    # Increased: 100 → 1000 for large dataset
        warmup_steps=100,                  # Increased: 50 → 100
        learning_rate=5e-5,                # Reduced: 1e-4 → 5e-5 for stability
        logging_steps=50,                  # More frequent logging
        save_strategy="steps",
        save_steps=100,                    # Save every 100 steps
        eval_steps=100,
        evaluation_strategy="steps",
        lr_scheduler_type="cosine",
        fp16=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        optim="adamw_torch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=5,                # Keep 5 best checkpoints
    )
    
    print("="*80)
    print("📊 TRAINING CONFIGURATION")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Training Steps: {training_args.max_steps:,}")
    print(f"Warmup Steps: {training_args.warmup_steps}")
    print(f"Learning Rate: {training_args.learning_rate}")
    print(f"Scheduler: {training_args.lr_scheduler_type}")
    print(f"LoRA Rank: {lora_config.r}")
    print(f"Total Training Samples: {len(train_dataset):,}")
    print(f"Output Directory: {training_args.output_dir}")
    print("="*80 + "\n")
    
    # Initialize model
    print("🚀 INITIALIZING MODEL...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="auto",
    )
    
    model.resize_token_embeddings(len(processor.tokenizer))
    model = get_peft_model(model, lora_config)
    print(f"✅ Model initialized with LoRA adapters\n")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("🚀 STARTING COMPREHENSIVE TRAINING...\n")
    trainer.train()
    
    # Save stats
    with open("training_comprehensive_stats.json", "w") as f:
        json.dump(dataset_stats, f, indent=2)
    
    print("\n✅ Training complete!")
    print(f"   Stats saved to: training_comprehensive_stats.json")
    print(f"   Model saved to: {training_args.output_dir}")
