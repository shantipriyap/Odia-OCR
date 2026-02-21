#!/usr/bin/env python3
"""
Fixed Odia OCR model evaluation - simpler approach
Tests if checkpoints can be loaded and generate any output
"""

import os
import sys
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
import torch
from PIL import Image
from pathlib import Path
import json

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_NAME = "shantipriya/odia-ocr-merged"
OUTPUT_DIR = "./qwen_odia_ocr_improved_v2"
CHECKPOINTS = ["checkpoint-50", "checkpoint-100", "checkpoint-150", "checkpoint-200", "checkpoint-250"]
TEST_SAMPLES = 20  # Reduced for faster testing

print("\n" + "="*70)
print("📊 ODIA OCR CHECKPOINT EVALUATION (Fixed)")
print("="*70)

# ============================================================================
# LOAD TEST DATASET
# ============================================================================

print("\n📥 Loading test dataset...")
try:
    dataset = load_dataset(DATASET_NAME, split="train")
    test_set = dataset.select(range(min(TEST_SAMPLES, len(dataset))))
    print(f"✅ Loaded {len(test_set)} test samples")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit(1)

# ============================================================================
# LOAD PROCESSOR
# ============================================================================

print("\n📦 Loading processor...")
try:
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("✅ Processor ready")
except Exception as e:
    print(f"❌ Error loading processor: {e}")
    exit(1)

# ============================================================================
# EVALUATE ONE SAMPLE WITH CHECKPOINT
# ============================================================================

def test_checkpoint(checkpoint_name):
    """Test if checkpoint can load and generate output"""
    checkpoint_path = f"{OUTPUT_DIR}/{checkpoint_name}"
    
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ {checkpoint_name}: Path not found")
        return None
    
    print(f"\n🔍 Testing {checkpoint_name}...", end=" ", flush=True)
    
    try:
        # Load base model
        print("[loading base]", end=" ", flush=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
        
        # Load LoRA adapter
        print("[loading lora]", end=" ", flush=True)
        model = PeftModel.from_pretrained(
            model,
            checkpoint_path,
            torch_dtype=torch.float16
        )
        model.eval()
        
        print("[testing]", end=" ", flush=True)
        
        # Try on first 3 samples
        predictions = []
        
        with torch.no_grad():
            for i, example in enumerate(test_set[:3]):
                try:
                    img = example["image"]
                    ref_text = str(example.get("text", "")).strip()
                    
                    if isinstance(img, str):
                        if os.path.exists(img):
                            img = Image.open(img).convert("RGB")
                        else:
                            continue
                    elif hasattr(img, "convert"):
                        img = img.convert("RGB")
                    else:
                        continue
                    
                    # Create simple prompt
                    prompt = "What text is in this image? Answer:"
                    
                    # Process inputs
                    inputs = processor(
                        images=img, 
                        text=prompt, 
                        return_tensors="pt",
                        padding=True
                    )
                    
                    # Move to device
                    for k in inputs:
                        if hasattr(inputs[k], 'to'):
                            inputs[k] = inputs[k].to(model.device)
                    
                    # Generate
                    output_ids = model.generate(
                        **inputs, 
                        max_new_tokens=50,
                        temperature=0.1,
                    )
                    
                    # Decode
                    generated_text = processor.decode(
                        output_ids[0][inputs["input_ids"].shape[1]:], 
                        skip_special_tokens=True
                    ).strip()
                    
                    if generated_text:
                        predictions.append({
                            "reference": ref_text[:30],
                            "generated": generated_text[:30],
                            "length": len(generated_text)
                        })
                
                except Exception as e:
                    continue
        
        # Report
        if predictions:
            print(f"✅ Generated {len(predictions)} predictions")
            for i, pred in enumerate(predictions[:2]):
                print(f"   [{i+1}] Ref: {pred['reference']}")
                print(f"       Gen: {pred['generated']}")
            
            return {
                "checkpoint": checkpoint_name,
                "status": "success",
                "predictions": len(predictions),
                "sample_predictions": predictions,
            }
        else:
            print(f"⚠️  No valid predictions")
            return None
            
    except Exception as e:
        print(f"❌ Error: {str(e)[:50]}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Free memory
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

# ============================================================================
# RUN TESTS
# ============================================================================

print("\n" + "="*70)
print("🚀 TESTING CHECKPOINTS")
print("="*70)

results = []
for checkpoint in CHECKPOINTS:
    result = test_checkpoint(checkpoint)
    if result:
        results.append(result)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("📈 SUMMARY")
print("="*70)

if results:
    print(f"\n✅ Successfully tested {len(results)} checkpoints:")
    for r in results:
        print(f"   • {r['checkpoint']}: {r['predictions']} predictions")
    
    # Save results
    with open(f"{OUTPUT_DIR}/checkpoint_tests.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to checkpoint_tests.json")
else:
    print(f"\n❌ Could not successfully test any checkpoints")

print("\n" + "="*70 + "\n")
