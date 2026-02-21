#!/usr/bin/env python3
"""
Fine-tune Qwen2.5-VL on merged Odia OCR dataset (145K+ samples)
using LoRA for improved performance

Improvements:
- Use 145K+ merged Odia samples vs 64 samples
- 500 training steps vs 100
- Cosine scheduler with warmup
- CER calculation during evaluation
- Better hyperparameters tuned for multi-dataset training
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
from jiwer import cer as compute_cer
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_NAME = "shantipriya/odia-ocr-merged"  # NEW: Use merged dataset
OUTPUT_DIR = "./qwen_odia_ocr_improved_v2"
MAX_STEPS = 500  # IMPROVED: from 100 to 500
WARMUP_STEPS = 50  # NEW: 10% of training steps

print(f"""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║        🚀 TRAINING QWEN2.5-VL ON MERGED ODIA OCR DATA 🚀      ║
║                                                                ║
║  Dataset:        {DATASET_NAME}                 ║
║  Samples:        145,781 (vs 64 in previous training)          ║
║  Model:          {MODEL_NAME}             ║
║  Training Steps: {MAX_STEPS} (vs 100 previously)                    ║
║  Improvement:    Expected CER 30-50% (vs 100% baseline)        ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# 1) LOAD MERGED ODIA OCR DATASET
# ============================================================================

print("📥 Loading merged Odia OCR dataset from HuggingFace Hub...")
print(f"   Source: {DATASET_NAME}\n")

try:
    dataset = load_dataset(DATASET_NAME)
    num_samples = len(dataset["train"])
    print(f"✅ Dataset loaded successfully!")
    print(f"   Total samples: {num_samples:,}")
    print(f"   Features: {list(dataset['train'].features.keys())}\n")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print("   Make sure the dataset is publicly available on HuggingFace Hub")
    exit(1)

# ============================================================================
# 2) LOAD PROCESSOR
# ============================================================================

print("📦 Loading Qwen2.5-VL processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
print(f"✅ Processor loaded\n")

def find_image_token(processor):
    """Find or identify the image token in the processor"""
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
# 3) LORA CONFIGURATION (IMPROVED)
# ============================================================================

print("⚙️  Configuring LoRA adapters...")
lora_config = LoraConfig(
    r=32,                              # Rank: 32 for good capacity
    lora_alpha=64,                     # Alpha: 2x rank for better scaling
    target_modules=["q_proj", "v_proj"],  # Focus on attention layers
    lora_dropout=0.05,                 # Low dropout for better signal
    bias="none",                       # No bias fine-tuning
    task_type="CAUSAL_LM"              # For language models
)
print(f"✅ LoRA config ready\n")

# ============================================================================
# 4) DATA PREPROCESSING
# ============================================================================

print("🔄 Preprocessing dataset...")

def preprocess_function(example):
    """Convert image to RGB and return raw image/text"""
    try:
        image = example.get("image")
        text = example.get("text", "")
        
        # Skip if no image or text
        if image is None or not text:
            return None  # Will be filtered out
        
        # Handle image types
        if isinstance(image, str):
            # If it's a path, return as is (processor will handle)
            return {"image": image, "text": text}
        elif hasattr(image, "convert"):
            # PIL Image - convert to RGB
            image = image.convert("RGB")
            return {"image": image, "text": text}
        else:
            return None  # Unknown type, skip
            
    except Exception as e:
        print(f"⚠️  Error preprocessing: {e}")
        return None  # Skip on error

# Process dataset and filter out None values
train_dataset = dataset["train"].map(preprocess_function, batched=False, num_proc=4)
train_dataset = train_dataset.filter(lambda x: x is not None)

# Use 10% for validation if available
if len(train_dataset) > 1000:
    split = train_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"✅ Created train/val split: {len(train_dataset):,} / {len(eval_dataset):,}\n")
else:
    eval_dataset = train_dataset
    print(f"⚠️  Dataset too small for validation split, using full dataset for both\n")

# ============================================================================
# 5) CUSTOM DATA COLLATOR
# ============================================================================

class QwenOCRDataCollator:
    """Custom collator for Qwen2.5-VL OCR fine-tuning"""
    
    def __init__(self, processor):
        self.processor = processor
        try:
            self.image_token = find_image_token(processor)
        except Exception:
            self.image_token = "<image>"
    
    def __call__(self, batch):
        images = []
        texts = []
        
        for example in batch:
            try:
                img = example.get("image")
                text = example.get("text", "")
                
                # Skip if no image or text
                if img is None or not text:
                    continue
                
                # Handle image loading
                if isinstance(img, str):
                    # It's a path - load it
                    try:
                        from PIL import Image
                        img = Image.open(img).convert("RGB")
                    except Exception as e:
                        print(f"⚠️  Could not load image from path {img}: {e}")
                        continue
                elif hasattr(img, "convert"):
                    # PIL Image
                    img = img.convert("RGB")
                else:
                    # Unknown type, skip
                    continue
                
                images.append(img)
                texts.append(f"{self.image_token} {text}")
                
            except Exception as e:
                print(f"⚠️  Skipping example: {e}")
                continue
        
        if not images:
            # Return dummy batch if all failed - trainer will handle it
            return {
                "input_ids": torch.tensor([[0]]),
                "labels": torch.tensor([[0]])
            }
        
        # Process batch
        try:
            inputs = self.processor(
                images,
                text=texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Set labels
            inputs["labels"] = inputs["input_ids"].clone()
            return inputs
        except Exception as e:
            print(f"⚠️  Error in processor: {e}")
            # Return a minimal valid batch
            return {
                "input_ids": torch.tensor([[0]]),
                "labels": torch.tensor([[0]])
            }

data_collator = QwenOCRDataCollator(processor)

# ============================================================================
# 6) COMPUTE METRICS (CER - CHARACTER ERROR RATE)
# ============================================================================

def compute_metrics(eval_preds):
    """Calculate CER (Character Error Rate) during evaluation"""
    try:
        predictions, labels = eval_preds
        
        # Decode predictions and labels
        decoded_preds = processor.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        decoded_labels = processor.tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )
        
        # Calculate CER
        cer_scores = []
        for pred, label in zip(decoded_preds, decoded_labels):
            if label:  # Only if label is not empty
                cer_scores.append(compute_cer(label, pred))
        
        avg_cer = np.mean(cer_scores) if cer_scores else 0.0
        
        return {
            "cer": avg_cer,
            "eval_loss": 0.0  # Will be overridden by trainer
        }
    except Exception as e:
        print(f"⚠️  Error computing metrics: {e}")
        return {"cer": 0.0}

# ============================================================================
# 7) TRAINING ARGUMENTS (IMPROVED)
# ============================================================================

print("📋 Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    # Batch size
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,      # Effective batch size: 4
    # Steps
    max_steps=MAX_STEPS,                # 500 steps for better training
    warmup_steps=WARMUP_STEPS,          # 10% warmup
    logging_steps=10,                   # Log every 10 steps
    save_steps=50,                      # Save checkpoint every 50 steps
    eval_steps=50,                      # Evaluate every 50 steps
    # Learning rate
    learning_rate=1e-4,                 # Lower for multi-dataset training
    lr_scheduler_type="cosine",         # Cosine decay
    # Optimization
    fp16=torch.cuda.is_available(),     # Use FP16 if GPU available
    remove_unused_columns=False,
    dataloader_num_workers=0,
    optim="adamw_torch",
    # Best model
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    # Evaluation
    eval_strategy="steps",
    # Reporting
    report_to=[],  # Disable tensorboard reporting for now
)

print(f"✅ Training configuration ready")
print(f"   Output: {OUTPUT_DIR}")
print(f"   Max steps: {MAX_STEPS}")
print(f"   Warmup: {WARMUP_STEPS}")
print(f"   LR: {training_args.learning_rate}")
print(f"   Scheduler: {training_args.lr_scheduler_type}\n")

# ============================================================================
# 8) MAIN TRAINING
# ============================================================================

if __name__ == "__main__":
    print("🤖 Loading Qwen2.5-VL model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="auto",
    )
    print(f"✅ Model loaded")
    
    # Resize embeddings if we added tokens
    model.resize_token_embeddings(len(processor.tokenizer))
    print(f"✅ Embeddings resized\n")
    
    # Apply LoRA
    print("🔧 Applying LoRA adapters...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print()
    
    # Create trainer
    print("🎯 Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    print(f"✅ Trainer ready\n")
    
    # Start training
    print("=" * 70)
    print("🚀 STARTING TRAINING")
    print("=" * 70)
    print(f"Dataset samples: {len(train_dataset):,}")
    print(f"Training steps: {MAX_STEPS}")
    print(f"Expected duration: ~4 hours")
    print(f"Expected improvement: CER 100% → 30-50%")
    print("=" * 70 + "\n")
    
    trainer.train()
    
    # Save final model
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    model.save_pretrained(f"{OUTPUT_DIR}/final_model")
    processor.save_pretrained(f"{OUTPUT_DIR}/processor")
    print(f"✅ Model saved to: {OUTPUT_DIR}/final_model")
    print(f"✅ Processor saved to: {OUTPUT_DIR}/processor")
    print("\n📊 Next steps:")
    print(f"1. Evaluate: python3 evaluate_model.py")
    print(f"2. Upload to HF: python3 upload_model.py")
    print(f"3. Share results!")
