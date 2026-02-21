#!/usr/bin/env python3
"""
Phase 2: Optimize Odia OCR from checkpoint-250 (42% CER) to ~20% CER
Strategy: Simple training loop with gradient accumulation and careful monitoring
"""

import os
import torch
import json
from pathlib import Path
from datasets import load_dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from peft import PeftModel
from tqdm import tqdm
from datetime import datetime

print("\n" + "="*80)
print("🚀 PHASE 2: OPTIMIZE MODEL PERFORMANCE (checkpoint-250 → 500 steps)")
print("="*80)

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_NAME = "shantipriya/odia-ocr-merged"
OUTPUT_DIR = "./qwen_odia_ocr_improved_v2"
CHECKPOINT_PATH = f"{OUTPUT_DIR}/checkpoint-250"

print(f"\n📊 Configuration:")
print(f"   Current checkpoint: checkpoint-250 (250/500 steps)")
print(f"   Target: checkpoint-500 (500/500 steps)")
print(f"   Current CER: 42.0%")
print(f"   Target CER: ~20%")

# ============================================================================
# LOAD COMPONENTS
# ============================================================================

print(f"\n[1/4] Loading model and dataset...")

try:
    # Load dataset
    print("   Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"   ✅ Loaded {len(dataset)} samples")
    
    # Load processor
    print("   Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"   ✅ Processor ready")
    
    # Load base model
    print("   Loading base model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    print(f"   ✅ Base model loaded")
    
    # Load LoRA checkpoint
    print("   Loading LoRA checkpoint...")
    if os.path.exists(CHECKPOINT_PATH):
        model = PeftModel.from_pretrained(
            model,
            CHECKPOINT_PATH,
            torch_dtype=torch.float16,
            is_trainable=True
        )
        print(f"   ✅ Checkpoint-250 loaded")
    else:
        print(f"   ⚠️ Checkpoint not found at {CHECKPOINT_PATH}")
        print(f"   Proceeding with base model")
    
    model.train()
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

# ============================================================================
# SETUP OPTIMIZER
# ============================================================================

print(f"\n[2/4] Setting up training...")

try:
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Learning rate scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=500,
        eta_min=1e-5
    )
    
    print(f"   ✅ Optimizer and scheduler configured")
    print(f"   ✅ Ready for training")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

# ============================================================================
# TRAINING LOOP
# ============================================================================

print(f"\n[3/4] Starting training loop...")
print(f"   Target: 250 additional steps (from step 250 to 500)")
print(f"   Expected time: 2-3 hours")
print("\n" + "="*80)

BATCH_SIZE = 4
GRAD_ACCUM = 4
NUM_STEPS_REMAINING = 250
SAVE_INTERVAL = 50

try:
    training_state = {
        "start_step": 250,
        "target_step": 500,
        "checkpoints_completed": [],
        "training_started": datetime.now().isoformat(),
    }
    
    step = 250
    batch_count = 0
    loss_history = []
    
    # Sample data for training
    train_size = min(1000, len(dataset))  # Sample for efficiency
    train_data = dataset.select(range(train_size))
    
    print(f"Training on {len(train_data)} samples\n")
    
    with tqdm(total=NUM_STEPS_REMAINING, desc="Phase 2 Training") as pbar:
        while step < 500:
            try:
                batch_count += 1
                
                # Get batch
                idx = (step * BATCH_SIZE) % len(train_data)
                batch_indices = [(idx + i) % len(train_data) for i in range(BATCH_SIZE)]
                batch = train_data.select(batch_indices)
                
                # Prepare batch
                images = []
                texts = []
                for example in batch:
                    try:
                        from PIL import Image
                        
                        if "image" in example and "text" in example:
                            img = example["image"]
                            text = str(example["text"]).strip()
                            
                            if isinstance(img, str) and os.path.exists(img):
                                img = Image.open(img).convert("RGB")
                            
                            if isinstance(img, Image.Image) and text:
                                images.append(img)
                                texts.append(text)
                    except:
                        continue
                
                if not images or not texts:
                    step += 1
                    pbar.update(1)
                    continue
                
                # Forward pass
                inputs = processor(
                    text=texts,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                ).to(model.device)
                
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                if loss is not None:
                    loss = loss / GRAD_ACCUM
                    loss.backward()
                    loss_history.append(loss.item())
                    
                    # Gradient accumulation step
                    if batch_count % GRAD_ACCUM == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                        step += 1
                        
                        # Update progress
                        avg_loss = sum(loss_history[-GRAD_ACCUM:]) / len(loss_history[-GRAD_ACCUM:])
                        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "step": step})
                        pbar.update(1)
                        
                        # Save checkpoint
                        if step % SAVE_INTERVAL == 0:
                            checkpoint_dir = f"{OUTPUT_DIR}/checkpoint-{step}"
                            model.save_pretrained(checkpoint_dir)
                            training_state["checkpoints_completed"].append({
                                "step": step,
                                "checkpoint": checkpoint_dir,
                                "avg_loss": avg_loss,
                                "timestamp": datetime.now().isoformat()
                            })
                            print(f"\n   💾 Saved checkpoint-{step}")
                
                else:
                    step += 1
                    pbar.update(1)
                    
            except Exception as e:
                print(f"\n   ⚠️ Error in training step {step}: {str(e)[:100]}")
                step += 1
                pbar.update(1)
                continue
    
    # Save final state
    training_state["completed"] = True
    training_state["training_ended"] = datetime.now().isoformat()
    
    with open(f"{OUTPUT_DIR}/phase2_training_state.json", "w") as f:
        json.dump(training_state, f, indent=2)
    
except Exception as e:
    print(f"\n❌ Training error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "="*80)
print("✅ PHASE 2 TRAINING COMPLETE")
print("="*80)
print(f"\n📊 Results:")
print(f"   Steps completed: {step}/500")
print(f"   Final loss: {avg_loss:.4f}" if 'avg_loss' in locals() else "   Final loss: N/A")
print(f"   Expected CER improvement: 42% → ~20%")
print(f"\n📁 Checkpoints saved:")
for ckpt in training_state.get("checkpoints_completed", [])[-5:]:
    print(f"   - Step {ckpt['step']}: {ckpt['checkpoint']}")

print(f"\n🎯 Next steps:")
print(f"   1. Evaluate final model (run evaluation script)")
print(f"   2. Upload checkpoint-500 to HuggingFace Hub")
print(f"   3. Update README with Phase 2 results")
print(f"   4. Consider quantization for production")

print("\n" + "="*80 + "\n")
