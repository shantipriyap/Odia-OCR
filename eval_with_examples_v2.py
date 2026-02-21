#!/usr/bin/env python3
"""
OCR Model Evaluation Script - Fixed Version
Evaluates fine-tuned Qwen2.5-VL model on Odia OCR task.
"""

import json
import os
import torch
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

# Configuration
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
CHECKPOINT_PATH = "./qwen_ocr_finetuned/checkpoint-50"
OUTPUT_DIR = "./eval_results"
NUM_SAMPLES = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def normalize_text(text):
    """Normalize Odia text for comparison."""
    if not text:
        return ""
    text = text.strip()
    # Unicode NFC normalization for Odia script
    import unicodedata
    return unicodedata.normalize('NFC', text)

def calculate_cer(reference, hypothesis):
    """Calculate Character Error Rate."""
    if not reference:
        return 1.0 if hypothesis else 0.0
    
    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, reference, hypothesis)
    matches = sum(block.size for block in matcher.get_matching_blocks())
    error_rate = 1.0 - (matches / len(reference))
    return max(0.0, min(1.0, error_rate))

def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if not ref_words:
        return 1.0 if hyp_words else 0.0
    
    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, ref_words, hyp_words)
    matches = sum(block.size for block in matcher.get_matching_blocks())
    error_rate = 1.0 - (matches / len(ref_words))
    return max(0.0, min(1.0, error_rate))

def load_model_and_processor():
    """Load model with PEFT adapters if available."""
    print("\n[1/5] Loading Model and Processor...")
    
    # Try to load with PEFT adapters
    try:
        print(f"  Loading processor from {MODEL_ID}...")
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        
        print(f"  Loading base model {MODEL_ID}...")
        # Use the model class from transformers.models for Qwen2.5-VL
        from transformers import Qwen2_5_VLForConditionalGeneration
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, 
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Try to load PEFT adapters
        if os.path.exists(CHECKPOINT_PATH):
            print(f"  Loading PEFT adapters from {CHECKPOINT_PATH}...")
            try:
                model = PeftModel.from_pretrained(model, CHECKPOINT_PATH)
                model = model.merge_and_unload()
                print("  ✓ PEFT adapters loaded and merged")
            except Exception as e:
                print(f"  ⚠ Could not load PEFT adapters: {e}")
                print("  Using base model only")
        else:
            print(f"  ✓ Using base model (checkpoint not found at {CHECKPOINT_PATH})")
        
        print("  ✓ Model and processor loaded successfully")
        return model, processor
    
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        raise

def load_dataset_samples():
    """Load Odia OCR dataset."""
    print("\n[2/5] Loading dataset...")
    try:
        dataset = load_dataset("OdiaGenAIOCR/Odia-lipi-ocr-data", split="train")
        print(f"  ✓ Dataset loaded: {len(dataset)} samples available")
        return dataset
    except Exception as e:
        print(f"  ✗ Error loading dataset: {e}")
        raise

def extract_text_from_response(response_text):
    """Extract Odia text from model response."""
    if not response_text:
        return ""
    
    response_text = str(response_text).strip()
    
    # Remove common prefixes/suffixes
    patterns = [
        r"^extracted.*?:\s*",
        r"^text.*?:\s*",
        r"^ocr.*?:\s*",
        r"\s*$"
    ]
    
    import re
    for pattern in patterns:
        response_text = re.sub(pattern, "", response_text, flags=re.IGNORECASE)
    
    return response_text.strip()

def run_ocr_inference(model, processor, image, max_retries=1):
    """Run OCR inference on an image."""
    try:
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Create OCR prompt
        prompt = "Extract all the text from this image. Provide only the extracted text."
        
        # Process image
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0
            )
        
        # Decode
        output_text = processor.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract text from response
        extracted_text = extract_text_from_response(output_text)
        
        return extracted_text
    
    except Exception as e:
        return ""

def evaluate_model(model, processor, dataset):
    """Evaluate model on dataset."""
    print(f"\n[3/5] Running OCR inference on {NUM_SAMPLES} samples...")
    
    results = []
    predictions = []
    timings = []
    
    for idx in tqdm(range(min(NUM_SAMPLES, len(dataset))), desc="Evaluating"):
        try:
            sample = dataset[idx]
            
            # Get image and reference text
            if "image" not in sample or "text" not in sample:
                continue
            
            image = sample["image"]
            reference_text = normalize_text(sample.get("text", ""))
            
            if not isinstance(image, Image.Image):
                if isinstance(image, dict) and "bytes" in image:
                    image = Image.open(BytesIO(image["bytes"]))
                else:
                    continue
            
            # Run inference
            start_time = time.time()
            predicted_text = run_ocr_inference(model, processor, image)
            elapsed_ms = (time.time() - start_time) * 1000
            
            predicted_text = normalize_text(predicted_text)
            
            # Calculate metrics
            cer = calculate_cer(reference_text, predicted_text)
            wer = calculate_wer(reference_text, predicted_text)
            exact_match = 1 if predicted_text == reference_text else 0
            
            # Store results
            result = {
                "index": idx,
                "predicted": predicted_text,
                "reference": reference_text,
                "inference_ms": elapsed_ms,
                "char_error_rate": cer,
                "word_error_rate": wer,
                "exact_match": exact_match
            }
            
            results.append(result)
            predictions.append({
                "index": idx,
                "predicted": predicted_text,
                "reference": reference_text
            })
            timings.append(elapsed_ms)
            
            # Try to save example images
            if idx < 5:
                try:
                    save_example_image(image, predicted_text, reference_text, idx)
                except Exception as e:
                    pass
        
        except Exception as e:
            pass
    
    return results, predictions, timings

def save_example_image(image, predicted, reference, idx):
    """Save example image with predictions."""
    try:
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # Add text to image
        y_offset = 10
        text_color = (255, 0, 0)  # Red
        
        # Try to use default font
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        text_lines = [
            f"Predicted: {predicted[:100]}",
            f"Reference: {reference[:100]}"
        ]
        
        for line in text_lines:
            draw.text((10, y_offset), line, fill=text_color, font=font)
            y_offset += 20
        
        # Save
        output_path = Path(OUTPUT_DIR) / f"example_{idx:03d}.jpg"
        img_copy.save(output_path)
    except:
        pass

def calculate_metrics(results):
    """Calculate aggregate metrics."""
    if not results:
        return None
    
    cer_values = [r["char_error_rate"] for r in results]
    wer_values = [r["word_error_rate"] for r in results]
    exact_matches = [r["exact_match"] for r in results]
    timings = [r["inference_ms"] for r in results]
    
    metrics = {
        "total_samples": len(results),
        "char_error_rate": {
            "mean": float(np.mean(cer_values)),
            "median": float(np.median(cer_values)),
            "std": float(np.std(cer_values)),
            "min": float(np.min(cer_values)),
            "max": float(np.max(cer_values))
        },
        "word_error_rate": {
            "mean": float(np.mean(wer_values)),
            "median": float(np.median(wer_values)),
            "std": float(np.std(wer_values)),
            "min": float(np.min(wer_values)),
            "max": float(np.max(wer_values))
        },
        "exact_match_accuracy": float(np.mean(exact_matches)),
        "inference_time_ms": {
            "mean": float(np.mean(timings)),
            "median": float(np.median(timings)),
            "min": float(np.min(timings)),
            "max": float(np.max(timings))
        }
    }
    
    return metrics

def generate_report(metrics, predictions):
    """Generate evaluation report."""
    print("\n[5/5] Generating report...")
    
    report = f"""# OCR Model Evaluation Report

**Model:** {MODEL_ID}
**Checkpoint:** {CHECKPOINT_PATH}
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Device:** {DEVICE}

## Summary Metrics

### Character Error Rate (CER)
- **Mean:** {metrics['char_error_rate']['mean']:.4f}
- **Median:** {metrics['char_error_rate']['median']:.4f}
- **Std Dev:** {metrics['char_error_rate']['std']:.4f}
- **Range:** [{metrics['char_error_rate']['min']:.4f}, {metrics['char_error_rate']['max']:.4f}]

### Word Error Rate (WER)
- **Mean:** {metrics['word_error_rate']['mean']:.4f}
- **Median:** {metrics['word_error_rate']['median']:.4f}
- **Std Dev:** {metrics['word_error_rate']['std']:.4f}
- **Range:** [{metrics['word_error_rate']['min']:.4f}, {metrics['word_error_rate']['max']:.4f}]

### Exact Match Accuracy
- **Accuracy:** {metrics['exact_match_accuracy']*100:.2f}%

### Inference Time
- **Mean:** {metrics['inference_time_ms']['mean']:.2f}ms
- **Median:** {metrics['inference_time_ms']['median']:.2f}ms
- **Min:** {metrics['inference_time_ms']['min']:.2f}ms
- **Max:** {metrics['inference_time_ms']['max']:.2f}ms

## Sample Predictions (First 5)

"""
    
    for i, pred in enumerate(predictions[:5]):
        report += f"\n### Sample {i+1}\n"
        report += f"- **Predicted:** {pred['predicted'][:200]}\n"
        report += f"- **Reference:** {pred['reference'][:200]}\n"
    
    # Save report
    report_path = Path(OUTPUT_DIR) / "EVAL_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"  ✓ Report saved to {report_path}")
    return report

def main():
    print("=" * 60)
    print("OCR Model Evaluation Script")
    print("=" * 60)
    
    try:
        # Load model
        model, processor = load_model_and_processor()
        
        # Load dataset
        dataset = load_dataset_samples()
        
        # Run evaluation
        results, predictions, timings = evaluate_model(model, processor, dataset)
        
        if not results:
            print("\n✗ No results generated!")
            return
        
        # Calculate metrics
        print(f"\n[4/5] Calculating metrics...")
        metrics = calculate_metrics(results)
        
        # Save metrics
        metrics_path = Path(OUTPUT_DIR) / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Metrics saved to {metrics_path}")
        
        # Save predictions
        pred_path = Path(OUTPUT_DIR) / "predictions.jsonl"
        with open(pred_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  ✓ Predictions saved to {pred_path}")
        
        # Generate report
        report = generate_report(metrics, predictions)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Results saved to: {OUTPUT_DIR}")
        print(f"  - metrics.json")
        print(f"  - predictions.jsonl")
        print(f"  - EVAL_REPORT.md")
        print(f"  - example_*.jpg (visual examples)")
        print("=" * 60)
        
        # Print key metrics
        print(f"\n📈 Key Metrics:")
        print(f"  CER (mean):     {metrics['char_error_rate']['mean']:.4f}")
        print(f"  WER (mean):     {metrics['word_error_rate']['mean']:.4f}")
        print(f"  Exact Match:    {metrics['exact_match_accuracy']*100:.2f}%")
        print(f"  Inference Time: {metrics['inference_time_ms']['mean']:.2f}ms avg")
        print()
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
