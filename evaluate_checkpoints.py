#!/usr/bin/env python3
"""
Evaluate Odia OCR model checkpoints for Character Error Rate (CER)
Measure accuracy improvement across training steps
"""

import os
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel
import torch
from jiwer import cer as compute_cer
from PIL import Image
from pathlib import Path
import json

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_NAME = "shantipriya/odia-ocr-merged"
OUTPUT_DIR = "./qwen_odia_ocr_improved_v2"
CHECKPOINTS = ["checkpoint-50", "checkpoint-100", "checkpoint-150", "checkpoint-200", "checkpoint-250"]
TEST_SAMPLES = 100  # Evaluate on 100 test samples

print("""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║         📊 ODIA OCR MODEL EVALUATION 📊                        ║
║     Measuring Character Error Rate across checkpoints          ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# LOAD TEST DATASET
# ============================================================================

print("📥 Loading test dataset...")
try:
    dataset = load_dataset(DATASET_NAME)
    test_set = dataset["train"].select(range(min(TEST_SAMPLES, len(dataset["train"]))))
    print(f"✅ Loaded {len(test_set)} test samples\n")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# Filter valid samples
def valid_sample(example):
    """Check if sample is valid"""
    try:
        img = example.get("image")
        text = str(example.get("text", "")).strip()
        
        if img is None or not text:
            return False
        
        if isinstance(img, str):
            return os.path.exists(img)
        return True
    except:
        return False

test_set = test_set.filter(valid_sample)
print(f"✅ Valid samples: {len(test_set)}\n")

# ============================================================================
# LOAD PROCESSOR
# ============================================================================

print("📦 Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("✅ Processor ready\n")

# ============================================================================
# EVALUATE CHECKPOINT
# ============================================================================

def evaluate_checkpoint(checkpoint_name):
    """Evaluate single checkpoint"""
    checkpoint_path = f"{OUTPUT_DIR}/{checkpoint_name}"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ {checkpoint_name}: Not found at {checkpoint_path}")
        return None
    
    print(f"🔍 Evaluating {checkpoint_name}...", end=" ", flush=True)
    
    try:
        # Load base model
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            trust_remote_code=True,
            device_map="auto",
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(
            model,
            checkpoint_path,
            torch_dtype="auto"
        )
        model.eval()
        
        cer_scores = []
        generated_texts = []
        reference_texts = []
        
        # Evaluate on test set
        with torch.no_grad():
            for i, example in enumerate(test_set):
                try:
                    # Load image
                    img = example["image"]
                    ref_text = str(example["text"]).strip()
                    
                    if isinstance(img, str):
                        if os.path.exists(img):
                            img = Image.open(img).convert("RGB")
                        else:
                            continue
                    elif hasattr(img, "convert"):
                        img = img.convert("RGB")
                    
                    # Generate
                    inputs = processor(img, text="Recognize Odia text:", return_tensors="pt")
                    
                    # Move to device
                    for k in inputs:
                        if hasattr(inputs[k], 'to'):
                            inputs[k] = inputs[k].to(model.device)
                    
                    output_ids = model.generate(**inputs, max_new_tokens=100)
                    gen_text = processor.decode(output_ids[0], skip_special_tokens=True)
                    
                    # Clean generated text (remove "Recognize Odia text:" prompt)
                    if "Recognize Odia text:" in gen_text:
                        gen_text = gen_text.split("Recognize Odia text:")[-1].strip()
                    
                    # Calculate CER
                    if ref_text and gen_text:
                        error_rate = compute_cer(ref_text, gen_text)
                        cer_scores.append(error_rate)
                        generated_texts.append(gen_text)
                        reference_texts.append(ref_text)
                    
                except Exception as e:
                    continue
        
        # Calculate statistics
        if cer_scores:
            avg_cer = sum(cer_scores) / len(cer_scores)
            min_cer = min(cer_scores)
            max_cer = max(cer_scores)
            
            # Count perfect matches (CER = 0)
            perfect = sum(1 for c in cer_scores if c == 0.0)
            
            print(f"✅")
            print(f"   Samples evaluated: {len(cer_scores)}")
            print(f"   Average CER: {avg_cer:.1%}")
            print(f"   Min CER: {min_cer:.1%}")
            print(f"   Max CER: {max_cer:.1%}")
            print(f"   Perfect matches: {perfect}/{len(cer_scores)}")
            
            # Show sample predictions
            print(f"\n   Sample predictions:")
            for i in range(min(3, len(generated_texts))):
                print(f"     Ref: {reference_texts[i][:40]}")
                print(f"     Gen: {generated_texts[i][:40]}")
                print(f"     CER: {cer_scores[i]:.1%}\n")
            
            return {
                "checkpoint": checkpoint_name,
                "avg_cer": avg_cer,
                "min_cer": min_cer,
                "max_cer": max_cer,
                "perfect_matches": perfect,
                "samples_evaluated": len(cer_scores),
            }
        else:
            print(f"⚠️  No valid predictions")
            return None
            
    except Exception as e:
        print(f"❌ Error: {str(e)[:60]}")
        return None
    finally:
        # Free memory
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

# ============================================================================
# EVALUATE ALL CHECKPOINTS
# ============================================================================

print("🚀 STARTING EVALUATIONS")
print("=" * 70 + "\n")

results = []
for checkpoint in CHECKPOINTS:
    result = evaluate_checkpoint(checkpoint)
    if result:
        results.append(result)
    print()

# ============================================================================
# SUMMARY REPORT
# ============================================================================

if results:
    print("\n" + "=" * 70)
    print("📊 EVALUATION SUMMARY")
    print("=" * 70)
    
    # Sort by checkpoint step
    results.sort(key=lambda x: int(x["checkpoint"].split("-")[1]))
    
    print(f"\n{'Checkpoint':<20} {'Avg CER':<12} {'Best':<12} {'Perfect':<10} {'Samples':<10}")
    print("-" * 70)
    
    for r in results:
        step = int(r["checkpoint"].split("-")[1])
        percent = (step / 500) * 100
        print(f"{r['checkpoint']:<20} {r['avg_cer']:>6.1%}      {r['min_cer']:>6.1%}      {r['perfect_matches']:>5}       {r['samples_evaluated']:>8}")
    
    # Calculate improvement
    first_cer = results[0]["avg_cer"]
    last_cer = results[-1]["avg_cer"]
    improvement = ((first_cer - last_cer) / first_cer * 100) if first_cer > 0 else 0
    
    print("\n" + "=" * 70)
    print(f"🎯 IMPROVEMENT")
    print("=" * 70)
    print(f"First checkpoint (50): {first_cer:.1%}")
    print(f"Last checkpoint (250):  {last_cer:.1%}")
    print(f"Improvement:            {improvement:+.1f}%")
    print(f"Reduction:              {first_cer - last_cer:.1%}")
    
    # Save results
    results_file = f"{OUTPUT_DIR}/evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅ Results saved to: {results_file}")
    
else:
    print("❌ No valid results to report")

print("\n" + "=" * 70)
print("✅ EVALUATION COMPLETE")
print("=" * 70)
