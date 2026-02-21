#!/usr/bin/env python3
"""
Comprehensive OCR evaluation with visual examples and performance metrics.
Generates: CER, WER, timing stats, and sample images with extracted text.
"""

import unicodedata
import torch
import time
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from jiwer import cer, wer
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# --------------------
# CONFIG
# --------------------
MODEL_PATH = "./qwen_ocr_finetuned"
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_NAME = "OdiaGenAIOCR/Odia-lipi-ocr-data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 256
NUM_SAMPLES = 50  # Evaluate on first N samples
OUTPUT_DIR = Path("./eval_results")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Output directory: {OUTPUT_DIR}")

# --------------------
# LOAD MODEL & PROCESSOR
# --------------------
print("\n[1/5] Loading model and processor...")
try:
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"✓ Processor loaded from {MODEL_PATH}")
except Exception as e:
    print(f"  Processor not found at {MODEL_PATH}, loading from base model: {e}")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"✓ Processor loaded from {MODEL_NAME}")

device_map = {"": "cuda:0"} if torch.cuda.is_available() else None
try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map=device_map,
    )
    print(f"✓ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"  Model not found at {MODEL_PATH}, loading base and applying PEFT adapters: {e}")
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map=device_map,
    )
    try:
        model = PeftModel.from_pretrained(base, MODEL_PATH, device_map=device_map)
        print(f"✓ PEFT adapters applied from {MODEL_PATH}")
    except:
        model = base
        print(f"✓ Using base model (no PEFT adapters found)")

model.eval()

# --------------------
# LOAD DATASET
# --------------------
print("\n[2/5] Loading dataset...")
try:
    dataset = load_dataset(DATASET_NAME)["train"]
    print(f"✓ Dataset loaded: {len(dataset)} samples available")
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    print("  Using synthetic data instead...")
    dataset = None

# --------------------
# NORMALIZATION
# --------------------
def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    return text

# --------------------
# OCR INFERENCE
# --------------------
def ocr_predict(image):
    start_time = time.time()
    inputs = processor(
        images=image.convert("RGB"),
        return_tensors="pt"
    )

    # move inputs to model device if model is on CUDA
    if device_map:
        target_device = device_map[""]
    else:
        target_device = DEVICE

    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS
        )

    pred = processor.decode(
        output_ids[0],
        skip_special_tokens=True
    )
    elapsed = time.time() - start_time
    return normalize_text(pred), elapsed

# --------------------
# GENERATE VISUAL EXAMPLES
# --------------------
def save_example_image(image, predicted_text, reference_text, index):
    """Save image with predictions overlaid."""
    img_copy = image.copy()
    if img_copy.mode != "RGB":
        img_copy = img_copy.convert("RGB")
    
    # Add text annotations
    draw = ImageDraw.Draw(img_copy)
    
    # Try to use a default font; fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    # Add predictions at bottom of image
    y_offset = img_copy.height + 20
    new_height = img_copy.height + 180
    result = Image.new("RGB", (img_copy.width, new_height), color=(255, 255, 255))
    result.paste(img_copy, (0, 0))
    
    draw = ImageDraw.Draw(result)
    draw.text((10, img_copy.height + 10), f"Predicted: {predicted_text[:80]}", fill=(0, 0, 0), font=font)
    draw.text((10, img_copy.height + 60), f"Reference: {reference_text[:80]}", fill=(100, 100, 100), font=font)
    
    output_path = OUTPUT_DIR / f"example_{index:03d}.jpg"
    result.save(output_path, quality=85)
    return str(output_path)

# --------------------
# RUN EVALUATION
# --------------------
print(f"\n[3/5] Running OCR inference on {NUM_SAMPLES} samples...")

predictions = []
references = []
timings = []
example_paths = []

if dataset is not None:
    samples = dataset.select(range(min(NUM_SAMPLES, len(dataset))))
    
    for idx, sample in enumerate(tqdm(samples, desc="Evaluating")):
        try:
            image = sample["image"]
            reference = normalize_text(sample.get("text", ""))
            
            predicted, elapsed = ocr_predict(image)
            
            predictions.append(predicted)
            references.append(reference)
            timings.append(elapsed)
            
            # Save first 5 examples
            if idx < 5:
                path = save_example_image(image, predicted, reference, idx)
                example_paths.append({
                    "index": idx,
                    "image": path,
                    "predicted": predicted[:100],
                    "reference": reference[:100],
                    "time_ms": elapsed * 1000
                })
                
        except Exception as e:
            print(f"  Error processing sample {idx}: {e}")
            continue

# --------------------
# CALCULATE METRICS
# --------------------
print("\n[4/5] Calculating metrics...")

if predictions and references:
    # Character Error Rate
    cer_score = cer(references, predictions)
    
    # Word Error Rate
    wer_score = wer(references, predictions)
    
    # Exact match accuracy
    exact_matches = sum(1 for p, r in zip(predictions, references) if p == r)
    accuracy = exact_matches / len(predictions) * 100 if predictions else 0
    
    # Timing statistics
    avg_time = sum(timings) / len(timings) if timings else 0
    min_time = min(timings) if timings else 0
    max_time = max(timings) if timings else 0
    
    metrics = {
        "model": MODEL_NAME,
        "fine_tuned_checkpoint": MODEL_PATH,
        "num_samples": len(predictions),
        "metrics": {
            "character_error_rate": round(cer_score, 4),
            "word_error_rate": round(wer_score, 4),
            "exact_match_accuracy": round(accuracy, 2),
        },
        "timing": {
            "avg_inference_ms": round(avg_time * 1000, 2),
            "min_inference_ms": round(min_time * 1000, 2),
            "max_inference_ms": round(max_time * 1000, 2),
        },
        "examples": example_paths
    }
    
    print(f"\n{'='*60}")
    print(f"PERFORMANCE STATISTICS (evaluated on {len(predictions)} samples)")
    print(f"{'='*60}")
    print(f"Character Error Rate (CER):    {cer_score:.2%}")
    print(f"Word Error Rate (WER):         {wer_score:.2%}")
    print(f"Exact Match Accuracy:          {accuracy:.2f}%")
    print(f"\nInference Timing:")
    print(f"  Average:  {avg_time*1000:.2f} ms")
    print(f"  Min:      {min_time*1000:.2f} ms")
    print(f"  Max:      {max_time*1000:.2f} ms")
    print(f"{'='*60}\n")
    
    # Save metrics to JSON
    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"✓ Metrics saved to {metrics_path}")
    
    # Save detailed predictions
    detailed_path = OUTPUT_DIR / "predictions.jsonl"
    with open(detailed_path, "w") as f:
        for i, (pred, ref, elapsed) in enumerate(zip(predictions, references, timings)):
            line = {
                "index": i,
                "predicted": pred,
                "reference": ref,
                "inference_ms": round(elapsed * 1000, 2),
                "char_error_rate": round(cer([ref], [pred]), 4),
                "word_error_rate": round(wer([ref], [pred]), 4),
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"✓ Detailed predictions saved to {detailed_path}")

# --------------------
# GENERATE REPORT
# --------------------
print("\n[5/5] Generating report...")

report_path = OUTPUT_DIR / "EVAL_REPORT.md"
with open(report_path, "w") as f:
    f.write("# OCR Model Evaluation Report\n\n")
    f.write(f"**Model:** {MODEL_NAME}\n")
    f.write(f"**Fine-tuned Checkpoint:** {MODEL_PATH}\n")
    f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    if predictions and references:
        f.write("## Performance Metrics\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Character Error Rate (CER) | {cer_score:.2%} |\n")
        f.write(f"| Word Error Rate (WER) | {wer_score:.2%} |\n")
        f.write(f"| Exact Match Accuracy | {accuracy:.2f}% |\n")
        f.write(f"| Samples Evaluated | {len(predictions)} |\n\n")
        
        f.write("## Inference Timing\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Average Inference Time | {avg_time*1000:.2f} ms |\n")
        f.write(f"| Min Inference Time | {min_time*1000:.2f} ms |\n")
        f.write(f"| Max Inference Time | {max_time*1000:.2f} ms |\n\n")
        
        if example_paths:
            f.write("## Example Predictions\n\n")
            for ex in example_paths:
                f.write(f"### Sample {ex['index'] + 1}\n")
                f.write(f"![Example Image]({Path(ex['image']).name})\n\n")
                f.write(f"**Predicted:** {ex['predicted']}\n\n")
                f.write(f"**Reference:** {ex['reference']}\n\n")
                f.write(f"**Inference Time:** {ex['time_ms']:.2f} ms\n\n")

print(f"✓ Report generated: {report_path}\n")

print(f"\n{'='*60}")
print(f"📊 EVALUATION COMPLETE")
print(f"{'='*60}")
print(f"Results saved to: {OUTPUT_DIR}")
print(f"  - metrics.json (summary stats)")
print(f"  - predictions.jsonl (detailed results)")
print(f"  - EVAL_REPORT.md (formatted report)")
print(f"  - example_*.jpg (visual examples)")
print(f"{'='*60}\n")
