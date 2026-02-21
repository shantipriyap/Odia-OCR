#!/usr/bin/env python3
"""
Phase 2: Complete Odia OCR Training from checkpoint-250 to 500 steps
Improves model from 42% CER (50% trained) to target ~20% CER (100% trained)
"""

import os
import sys
import torch
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_NAME = "shantipriya/odia-ocr-merged"
OUTPUT_DIR = "./qwen_odia_ocr_improved_v2"
CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoint-250"  # Resume from here
PHASE2_START_STEP = 250
MAX_STEPS = 500
SAVE_STEPS = 50

print("\n" + "="*80)
print("🚀 PHASE 2: IMPROVE MODEL PERFORMANCE")
print("="*80)
print(f"\n📊 Configuration:")
print(f"   Starting from: {CHECKPOINT_DIR}")
print(f"   Start step: {PHASE2_START_STEP}")
print(f"   Target steps: {MAX_STEPS}")
print(f"   Phase 2 steps: {MAX_STEPS - PHASE2_START_STEP}")
print(f"   Current CER: 42.0%")
print(f"   Target CER: ~20%")
print(f"   Expected improvement: 50-55%")

# ============================================================================
# STEP 1: LOAD DATASET
# ============================================================================

print(f"\n[1/5] 📥 Loading dataset...")
try:
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"✅ Loaded {len(dataset)} samples from {DATASET_NAME}")
    
    # Create train/val split
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"   Training set: {len(train_dataset)} samples")
    print(f"   Validation set: {len(eval_dataset)} samples")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    sys.exit(1)

# ============================================================================
# STEP 2: LOAD BASE MODEL & PROCESSOR
# ============================================================================

print(f"\n[2/5] 🤖 Loading model and processor...")
try:
    # Load processor
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"✅ Processor loaded")
    
    # Load base model
    print(f"   Loading base model: {MODEL_NAME}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    print(f"✅ Base model loaded ({model.config.hidden_size})")
    
    # Load checkpoint-250 (LoRA adapter)
    print(f"   Loading LoRA checkpoint: {CHECKPOINT_DIR}...")
    if os.path.exists(CHECKPOINT_DIR):
        model = PeftModel.from_pretrained(
            model,
            CHECKPOINT_DIR,
            torch_dtype=torch.float16,
            is_trainable=True
        )
        print(f"✅ LoRA checkpoint loaded")
        print(f"   Training from step: {PHASE2_START_STEP}")
    else:
        print(f"❌ Checkpoint not found: {CHECKPOINT_DIR}")
        sys.exit(1)

except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 3: PREPARE TRAINING ARGUMENTS
# ============================================================================

print(f"\n[3/5] ⚙️  Preparing training configuration...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    warmup_steps=0,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    Save_strategy="steps",
    save_steps=SAVE_STEPS,
    logging_steps=10,
    logging_dir="./logs",
    report_to=["tensorboard"],
    seed=42,
    fp16=True,
    dataloader_num_workers=0,
    remove_unused_columns=False,
    optim="adamw_torch",
    gradient_checkpointing=True,
)

print(f"✅ Training configuration prepared:")
print(f"   Max steps: {MAX_STEPS}")
print(f"   Save steps: {SAVE_STEPS}")
print(f"   Learning rate: {training_args.learning_rate}")
print(f"   Batch size (effective): {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

# ============================================================================
# STEP 4: DATA COLLATOR & TRAINER
# ============================================================================

print(f"\n[4/5] 🏋️  Setting up trainer...")

def collate_fn(batch):
    """Prepare batch data for model"""
    images = []
    texts = []
    
    for example in batch:
        try:
            from PIL import Image
            
            # Get image
            if "image" in example:
                img = example["image"]
                if isinstance(img, str) and os.path.exists(img):
                    img = Image.open(img).convert("RGB")
                
                if isinstance(img, Image.Image):
                    images.append(img)
                else:
                    continue
            else:
                continue
            
            # Get text
            text = str(example.get("text", "")).strip()
            if text:
                texts.append(text)
            else:
                images.pop()
                continue
                
        except:
            continue
    
    if not images or not texts:
        return None
    
    # Process with processor
    inputs = processor(
        images=images,
        text=texts,
        return_tensors="pt",
        padding=True,
    )
    
    return inputs

# Custom trainer to handle collate_fn
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    processing_class=processor,
)

print(f"✅ Trainer initialized")

# ============================================================================
# STEP 5: TRAIN MODEL
# ============================================================================

print(f"\n[5/5] 🚀 Starting Phase 2 training...")
print(f"   Target: Step {MAX_STEPS} (from {PHASE2_START_STEP})")
print(f"   Expected time: 2-3 hours on RTX A6000")
print(f"   Expected improvement: CER 42% → ~20%")
print("\n" + "="*80)

try:
    # Start training
    train_result = trainer.train()
    
    print("\n" + "="*80)
    print("✅ PHASE 2 TRAINING COMPLETE")
    print("="*80)
    print(f"\n📊 Training Results:")
    print(f"   Final step: {train_result.global_step}")
    print(f"   Final loss: {train_result.training_loss:.4f}")
    print(f"   Training time: {train_result.total_flos / (1e9 * 3600):.2f} hours")
    
    # Save final model
    print(f"\n💾 Saving final model...")
    final_checkpoint = f"{OUTPUT_DIR}/checkpoint-final"
    trainer.save_model(final_checkpoint)
    print(f"✅ Final model saved to {final_checkpoint}")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Evaluate final model accuracy (expected ~20% CER)")
    print(f"   2. Upload final checkpoint to HuggingFace Hub")
    print(f"   3. Consider quantization for production deployment")
    print(f"   4. Document improvements in README")
    
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print(f"Training timestamp: {datetime.now().isoformat()}")
print("="*80 + "\n")
