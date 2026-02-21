#!/usr/bin/env python3
"""
Odia OCR Model Accuracy Evaluation
Evaluates checkpoint-250 on the test dataset and calculates metrics
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset, Dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from PIL import Image, ImageDraw
import io
from jiwer import cer, wer

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_NAME = "shantipriya/odia-ocr-merged"
OUTPUT_DIR = "./qwen_odia_ocr_improved_v2"
CHECKPOINT = "checkpoint-250"
TEST_SIZE = 50  # Evaluate on 50 samples

print("\n" + "="*80)
print("📊 ODIA OCR MODEL ACCURACY EVALUATION")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATASET
# ============================================================================

print(f"\n[1/5] 📥 Loading test dataset ({TEST_SIZE} samples)...")
try:
    dataset = load_dataset(DATASET_NAME, split="train")
    
    # Create test split
    if len(dataset) > TEST_SIZE:
        indices = np.random.choice(len(dataset), TEST_SIZE, replace=False)
        test_dataset = dataset.select(indices.tolist())
    else:
        test_dataset = dataset
    
    print(f"✅ Loaded {len(test_dataset)} test samples")
    
    # Show sample
    sample = test_dataset[0]
    print(f"   Sample keys: {list(sample.keys())}")
    if "text" in sample:
        text_preview = str(sample["text"])[:50]
        print(f"   Sample text: {text_preview}...")

except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    sys.exit(1)

# ============================================================================
# STEP 2: LOAD MODEL & PROCESSOR
# ============================================================================

print(f"\n[2/5] 📦 Loading model and processor...")
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
    print(f"✅ Base model loaded")
    
    # Load LoRA adapter
    checkpoint_path = f"{OUTPUT_DIR}/{CHECKPOINT}"
    if os.path.exists(checkpoint_path):
        print(f"   Loading LoRA adapter: {checkpoint_path}...")
        model = PeftModel.from_pretrained(
            model,
            checkpoint_path,
            torch_dtype=torch.float16
        )
        print(f"✅ LoRA adapter loaded")
    else:
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print(f"   Using base model only")
    
    model.eval()
    print(f"✅ Model ready for inference")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 3: PREPARE TEST DATA
# ============================================================================

print(f"\n[3/5] 🔄 Preparing test samples...")

test_samples = []
for i, example in enumerate(test_dataset):
    try:
        # Get image
        if "image" in example:
            img = example["image"]
            if isinstance(img, str) and os.path.exists(img):
                img = Image.open(img).convert("RGB")
            elif not isinstance(img, Image.Image):
                # Try to load from path
                continue
        else:
            continue
        
        # Get reference text
        ref_text = str(example.get("text", "")).strip()
        if not ref_text or len(ref_text) < 2:
            continue
        
        test_samples.append({
            "image": img,
            "reference": ref_text,
            "index": i
        })
        
        if len(test_samples) >= TEST_SIZE:
            break
            
    except Exception as e:
        continue

print(f"✅ Prepared {len(test_samples)} valid test samples")

if len(test_samples) == 0:
    print("❌ No valid test samples found!")
    sys.exit(1)

# ============================================================================
# STEP 4: RUN INFERENCE
# ============================================================================

print(f"\n[4/5] 🧠 Running inference on {len(test_samples)} samples...")

predictions = []
references = []
errors = []
inference_times = []

import time

with torch.no_grad():
    for idx, sample in enumerate(test_samples):
        try:
            img = sample["image"]
            ref_text = sample["reference"]
            
            # Prepare input
            prompt = "Extract and read all the Odia text from this image. Return only the text content."
            
            # Process image
            inputs = processor(
                images=img,
                text=prompt,
                return_tensors="pt"
            ).to(model.device)
            
            # Generate
            start_time = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                num_beams=1,
                temperature=0.7,
                top_p=0.9,
            )
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Decode
            pred_text = processor.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up prediction
            pred_text = pred_text.strip()
            if len(pred_text) > len(prompt):
                # Try to remove prompt from output
                if pred_text.startswith(prompt):
                    pred_text = pred_text[len(prompt):].strip()
            
            predictions.append(pred_text)
            references.append(ref_text)
            
            # Show progress
            if (idx + 1) % 10 == 0:
                print(f"   Processed {idx + 1}/{len(test_samples)} samples ({inference_time:.2f}s per sample)")
            
        except Exception as e:
            print(f"   ⚠️  Error on sample {idx}: {str(e)[:50]}")
            errors.append({"sample": idx, "error": str(e)})
            continue

print(f"✅ Inference complete - {len(predictions)} predictions generated")

# ============================================================================
# STEP 5: CALCULATE METRICS
# ============================================================================

print(f"\n[5/5] 📈 Calculating accuracy metrics...")

if len(predictions) == 0 or len(references) == 0:
    print("❌ No predictions generated!")
    sys.exit(1)

# Character Error Rate
try:
    char_error_rate = cer(references, predictions)
    print(f"✅ Character Error Rate (CER): {char_error_rate:.2%}")
except:
    char_error_rate = None
    print(f"⚠️  CER calculation failed")

# Word Error Rate
try:
    word_error_rate = wer(references, predictions)
    print(f"✅ Word Error Rate (WER): {word_error_rate:.2%}")
except:
    word_error_rate = None
    print(f"⚠️  WER calculation failed")

# Match Rate
try:
    match_rate = sum(1 for r, p in zip(references, predictions) if r.lower().strip() == p.lower().strip()) / len(predictions)
    print(f"✅ Exact Match Rate: {match_rate:.2%}")
except:
    match_rate = None

# Character accuracy
try:
    char_accuracy = 1 - (char_error_rate if char_error_rate else 0)
    print(f"✅ Character Accuracy: {char_accuracy:.2%}")
except:
    char_accuracy = None

# Average inference time
avg_inference_time = np.mean(inference_times) if inference_times else 0
print(f"✅ Average Inference Time: {avg_inference_time:.3f}s per sample")

# ============================================================================
# DETAILED RESULTS
# ============================================================================

print(f"\n" + "="*80)
print("DETAILED EVALUATION RESULTS")
print("="*80)

print(f"\n📊 Metrics Summary:")
print(f"   Total Samples Evaluated: {len(predictions)}")
print(f"   Successful Predictions: {len(predictions)}")
print(f"   Failed/Incomplete: {len(test_samples) - len(predictions)}")

if char_error_rate is not None:
    print(f"\n🎯 Accuracy Metrics:")
    print(f"   Character Error Rate (CER): {char_error_rate:.4f} ({char_error_rate:.2%})")
    print(f"   Character Accuracy: {(1-char_error_rate):.4f} ({(1-char_error_rate):.2%})")

if word_error_rate is not None:
    print(f"   Word Error Rate (WER): {word_error_rate:.4f} ({word_error_rate:.2%})")

if match_rate is not None:
    print(f"   Exact Match Rate: {match_rate:.4f} ({match_rate:.2%})")

print(f"\n⚡ Performance Metrics:")
print(f"   Average Inference Time: {avg_inference_time:.3f}s")
print(f"   Min Inference Time: {min(inference_times):.3f}s")
print(f"   Max Inference Time: {max(inference_times):.3f}s")
print(f"   Throughput: {1/avg_inference_time:.2f} samples/second")

# Show sample predictions
print(f"\n📝 Sample Predictions (First 5):")
print("-" * 80)
for i in range(min(5, len(predictions))):
    print(f"\nSample {i+1}:")
    print(f"  Reference: {references[i][:100]}")
    print(f"  Prediction: {predictions[i][:100]}")
    if references[i].lower().strip() == predictions[i].lower().strip():
        print(f"  Status: ✅ MATCH")
    else:
        print(f"  Status: ❌ MISMATCH")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print(f"\n💾 Saving results...")

results = {
    "model": CHECKPOINT,
    "dataset": DATASET_NAME,
    "num_samples_evaluated": len(predictions),
    "metrics": {
        "character_error_rate": float(char_error_rate) if char_error_rate else None,
        "word_error_rate": float(word_error_rate) if word_error_rate else None,
        "character_accuracy": float(char_accuracy) if char_accuracy else None,
        "exact_match_rate": float(match_rate) if match_rate else None,
    },
    "performance": {
        "avg_inference_time_seconds": float(avg_inference_time),
        "min_inference_time_seconds": float(min(inference_times)) if inference_times else None,
        "max_inference_time_seconds": float(max(inference_times)) if inference_times else None,
        "throughput_samples_per_second": float(1/avg_inference_time) if avg_inference_time > 0 else None,
    },
    "samples": [
        {
            "reference": ref,
            "prediction": pred,
            "match": ref.lower().strip() == pred.lower().strip()
        }
        for ref, pred in zip(references, predictions)
    ]
}

# Save to JSON
output_file = "evaluation_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"✅ Results saved to {output_file}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"\n" + "="*80)
print("✅ EVALUATION COMPLETE")
print("="*80)
print(f"\n📍 Summary:")
print(f"   Model: {CHECKPOINT}")
print(f"   Test Samples: {len(predictions)}")
if char_error_rate is not None:
    print(f"   Character Accuracy: {(1-char_error_rate):.2%}")
if match_rate is not None:
    print(f"   Exact Match Rate: {match_rate:.2%}")
print(f"   Inference Speed: {1/avg_inference_time:.2f} samples/sec")
print(f"\n📄 Results saved to: {output_file}")
print("="*80 + "\n")
