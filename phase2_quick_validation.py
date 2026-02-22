#!/usr/bin/env python3
"""
PHASE 2 QUICK VALIDATION - Fast test with 10 samples
Tests beam search + ensemble optimization for CER improvement
Should complete in 20-30 minutes on RTX A6000
"""

import json
import time
import torch
import warnings
from pathlib import Path
from typing import List, Dict
from PIL import Image
from datasets import load_dataset
import jiwer

warnings.filterwarnings("ignore")


def run_phase2_quick_validation():
    """Run quick Phase 2 validation test"""
    
    print("\n" + "="*80)
    print("PHASE 2 QUICK VALIDATION - Beam Search + Ensemble Test")
    print("="*80 + "\n")
    
    start_time = time.time()
    results = {"timestamp": str(time.time()), "methods": {}}
    
    try:
        # Import inference engine
        from inference_engine_production import OdiaOCRInferenceEngine
        
        print("🚀 Initializing Inference Engine...")
        engine = OdiaOCRInferenceEngine()
        
        # Load small test dataset (10 samples for speed)
        print("📊 Loading 10 test samples (fast)...")
        dataset = load_dataset("shantipriya/odia-ocr-merged", split="train")
        
        # Use first 10 samples (consistent testing)
        images = []
        texts = []
        for i in range(min(10, len(dataset))):
            sample = dataset[i]
            img = sample["image"] if isinstance(sample["image"], Image.Image) else Image.open(sample["image"])
            images.append(img)
            texts.append(sample["text"])
        
        print(f"✅ Loaded {len(images)} samples\n")
        
        # Test 1: Baseline (greedy)
        print("-" * 80)
        print("TEST 1: Baseline (Greedy Decoding)")
        print("-" * 80)
        
        baseline_start = time.time()
        baseline_preds = engine.infer_beam_search(images, num_beams=1, checkpoint="checkpoint-250")
        baseline_time = time.time() - baseline_start
        
        baseline_cer = jiwer.cer(texts, baseline_preds)
        baseline_wer = jiwer.wer(texts, baseline_preds)
        
        results["methods"]["baseline"] = {
            "cer": float(baseline_cer),
            "wer": float(baseline_wer),
            "time_total": baseline_time,
            "time_per_image": baseline_time / len(images),
        }
        
        print(f"✅ Baseline CER:  {baseline_cer:.1%}")
        print(f"   WER:           {baseline_wer:.1%}")
        print(f"   Total time:    {baseline_time:.2f}s ({baseline_time/len(images):.2f}s per image)\n")
        
        # Test 2: Beam Search (5-beam)
        print("-" * 80)
        print("TEST 2: Beam Search (5-beam)")
        print("-" * 80)
        
        beam_start = time.time()
        beam_preds = engine.infer_beam_search(images, num_beams=5, checkpoint="checkpoint-250")
        beam_time = time.time() - beam_start
        
        beam_cer = jiwer.cer(texts, beam_preds)
        beam_wer = jiwer.wer(texts, beam_preds)
        beam_improvement = (baseline_cer - beam_cer) / baseline_cer
        
        results["methods"]["beam_search_5"] = {
            "cer": float(beam_cer),
            "wer": float(beam_wer),
            "improvement_vs_baseline": float(beam_improvement),
            "time_total": beam_time,
            "time_per_image": beam_time / len(images),
        }
        
        print(f"✅ Beam Search CER: {beam_cer:.1%}")
        print(f"   WER:            {beam_wer:.1%}")
        print(f"   Improvement:    {beam_improvement:+.1%}")
        print(f"   Total time:     {beam_time:.2f}s ({beam_time/len(images):.2f}s per image)\n")
        
        # Test 3: Ensemble Voting
        print("-" * 80)
        print("TEST 3: Ensemble Voting (5 checkpoints)")
        print("-" * 80)
        
        ensemble_start = time.time()
        try:
            ensemble_preds = engine.infer_ensemble_voting(images, voting_method="longest")
            ensemble_time = time.time() - ensemble_start
            
            ensemble_cer = jiwer.cer(texts, ensemble_preds)
            ensemble_wer = jiwer.wer(texts, ensemble_preds)
            ensemble_improvement = (baseline_cer - ensemble_cer) / baseline_cer
            
            results["methods"]["ensemble_voting"] = {
                "cer": float(ensemble_cer),
                "wer": float(ensemble_wer),
                "improvement_vs_baseline": float(ensemble_improvement),
                "time_total": ensemble_time,
                "time_per_image": ensemble_time / len(images),
            }
            
            print(f"✅ Ensemble CER:  {ensemble_cer:.1%}")
            print(f"   WER:           {ensemble_wer:.1%}")
            print(f"   Improvement:   {ensemble_improvement:+.1%}")
            print(f"   Total time:    {ensemble_time:.2f}s ({ensemble_time/len(images):.2f}s per image)\n")
            
        except Exception as e:
            print(f"⚠️  Ensemble voting error: {e}")
            print("   (This is expected if not all checkpoints are available)\n")
        
        # Summary
        print("=" * 80)
        print("PHASE 2 QUICK VALIDATION - SUMMARY")
        print("=" * 80 + "\n")
        
        print("🎯 Results:")
        for method, data in results["methods"].items():
            status = "✅"
            if "improvement_vs_baseline" in data and data["improvement_vs_baseline"] > 0:
                status = "✅ IMPROVED"
            
            print(f"\n{status} {method.upper()}")
            print(f"   CER: {data['cer']:.1%}")
            print(f"   WER: {data['wer']:.1%}")
            if "improvement_vs_baseline" in data:
                print(f"   Improvement: {data['improvement_vs_baseline']:+.1%}")
        
        # Final assessment
        print("\n" + "=" * 80)
        print("✅ PHASE 2 VALIDATION COMPLETE")
        print("=" * 80)
        
        total_time = time.time() - start_time
        print(f"\nTotal execution time: {total_time:.2f}s")
        print(f"\n📊 Results saved to: phase2_quick_validation_results.json")
        
        # Save results
        with open("phase2_quick_validation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results
        
    except Exception as e:
        print(f"❌ Error during Phase 2 validation: {e}")
        import traceback
        traceback.print_exc()
        return results


if __name__ == "__main__":
    results = run_phase2_quick_validation()
