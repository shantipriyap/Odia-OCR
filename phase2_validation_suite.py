#!/usr/bin/env python3
"""
PHASE 2 VALIDATION: Complete Testing & Benchmarking Suite

This script validates all optimization techniques and provides comprehensive
performance metrics for:
1. Baseline inference (greedy decoding)
2. Beam search variants
3. Ensemble voting approaches
4. Temperature sampling
5. Inference time & throughput analysis
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
from datasets import load_dataset
import jiwer
import warnings

warnings.filterwarnings("ignore")


class OCRBenchmarkSuite:
    """Comprehensive benchmarking for OCR optimization techniques"""
    
    def __init__(self):
        """Initialize benchmark suite"""
        self.results = {
            "timestamp": str(time.time()),
            "methods": {},
            "comparisons": {},
            "recommendations": {}
        }
        self.all_predictions = {}
    
    def prepare_test_dataset(self, num_samples: int = 50) -> Dict:
        """Load test dataset from HF Hub"""
        print(f"📊 Loading {num_samples} test samples...")
        
        dataset = load_dataset("shantipriya/odia-ocr-merged", split="train")
        
        # Randomly sample
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        
        images = []
        texts = []
        
        for idx in indices:
            sample = dataset[idx]
            img = sample["image"] if isinstance(sample["image"], Image.Image) else Image.open(sample["image"])
            images.append(img)
            texts.append(sample["text"])
        
        print(f"✅ Loaded {len(images)} samples")
        return {"images": images, "texts": texts}
    
    def calculate_metrics(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict:
        """Calculate CER, WER, and other metrics"""
        
        cer_scores = []
        wer_scores = []
        exact_matches = 0
        
        for pred, ref in zip(predictions, references):
            cer = jiwer.cer(ref, pred)
            wer = jiwer.wer(ref, pred)
            
            cer_scores.append(cer)
            wer_scores.append(wer)
            
            if pred.strip() == ref.strip():
                exact_matches += 1
        
        return {
            "cer": float(np.mean(cer_scores)),
            "cer_std": float(np.std(cer_scores)),
            "wer": float(np.mean(wer_scores)),
            "exact_match_rate": exact_matches / len(predictions),
            "num_samples": len(predictions),
        }
    
    def benchmark_method(
        self,
        method_name: str,
        inference_func,
        images: List[Image.Image],
        texts: List[str],
        **kwargs
    ) -> Dict:
        """Benchmark a single inference method"""
        
        print(f"\n🧪 Testing: {method_name}")
        
        start_time = time.time()
        predictions = inference_func(images, **kwargs)
        inference_time = time.time() - start_time
        
        metrics = self.calculate_metrics(predictions, texts)
        
        result = {
            "method": method_name,
            "total_time": inference_time,
            "time_per_image": inference_time / len(images),
            "throughput": len(images) / inference_time,
            "predictions_sample": [p[:100] for p in predictions[:3]],
            **metrics
        }
        
        print(f"   CER: {result['cer']:.1%} (±{result['cer_std']:.1%})")
        print(f"   WER: {result['wer']:.1%}")
        print(f"   Time: {inference_time:.2f}s total ({result['time_per_image']:.2f}s/img)")
        print(f"   Throughput: {result['throughput']:.2f} img/s")
        
        return result
    
    def run_full_benchmark(self, num_samples: int = 50) -> Dict:
        """Run complete benchmark suite"""
        
        print("\n" + "="*80)
        print("PHASE 2 OPTIMIZATION BENCHMARK SUITE")
        print("="*80)
        
        # Prepare data
        test_data = self.prepare_test_dataset(num_samples)
        images = test_data["images"]
        texts = test_data["texts"]
        
        # Import inference engine
        try:
            from inference_engine_production import OdiaOCRInferenceEngine
            engine = OdiaOCRInferenceEngine()
        except Exception as e:
            print(f"❌ Failed to load inference engine: {e}")
            return self.results
        
        # Test 1: Baseline (Greedy Decoding)
        print("\n" + "-"*80)
        print("METHOD 1: BASELINE (Greedy Decoding)")
        print("-"*80)
        
        baseline_result = self.benchmark_method(
            "Baseline - Greedy Decoding",
            engine.infer_beam_search,
            images,
            texts,
            num_beams=1,
            checkpoint="checkpoint-250"
        )
        self.results["methods"]["baseline"] = baseline_result
        
        # Test 2: Beam Search (3-beam)
        print("\n" + "-"*80)
        print("METHOD 2: BEAM SEARCH (3-beam)")
        print("-"*80)
        
        beam3_result = self.benchmark_method(
            "Beam Search - 3 beams",
            engine.infer_beam_search,
            images,
            texts,
            num_beams=3,
            checkpoint="checkpoint-250"
        )
        self.results["methods"]["beam_search_3"] = beam3_result
        
        # Test 3: Beam Search (5-beam)
        print("\n" + "-"*80)
        print("METHOD 3: BEAM SEARCH (5-beam)")
        print("-"*80)
        
        beam5_result = self.benchmark_method(
            "Beam Search - 5 beams",
            engine.infer_beam_search,
            images,
            texts,
            num_beams=5,
            checkpoint="checkpoint-250"
        )
        self.results["methods"]["beam_search_5"] = beam5_result
        
        # Test 4: Beam Search (7-beam)
        print("\n" + "-"*80)
        print("METHOD 4: BEAM SEARCH (7-beam)")
        print("-"*80)
        
        beam7_result = self.benchmark_method(
            "Beam Search - 7 beams",
            engine.infer_beam_search,
            images,
            texts,
            num_beams=7,
            checkpoint="checkpoint-250"
        )
        self.results["methods"]["beam_search_7"] = beam7_result
        
        # Test 5: Ensemble Voting
        print("\n" + "-"*80)
        print("METHOD 5: ENSEMBLE VOTING (5 checkpoints)")
        print("-"*80)
        
        ensemble_result = self.benchmark_method(
            "Ensemble Voting - All checkpoints",
            engine.infer_ensemble_voting,
            images,
            texts,
            voting_method="longest"
        )
        self.results["methods"]["ensemble_voting"] = ensemble_result
        
        # Test 6: Temperature Sampling (0.7)
        print("\n" + "-"*80)
        print("METHOD 6: TEMPERATURE SAMPLING (T=0.7)")
        print("-"*80)
        
        temp_result = self.benchmark_method(
            "Temperature Sampling - T=0.7",
            engine.infer_temperature_sampling,
            images,
            texts,
            temperature=0.7,
            checkpoint="checkpoint-250"
        )
        self.results["methods"]["temperature_07"] = temp_result
        
        # Generate Comparisons
        print("\n" + "="*80)
        print("COMPARISON & ANALYSIS")
        print("="*80)
        
        self._analyze_results()
        
        return self.results
    
    def _analyze_results(self):
        """Analyze and compare results"""
        
        if not self.results["methods"]:
            print("❌ No results to analyze")
            return
        
        # CER Comparison
        print("\n📊 CHARACTER ERROR RATE (CER) Comparison")
        print("-" * 60)
        
        baseline_cer = self.results["methods"]["baseline"]["cer"]
        
        method_cers = []
        for method_name, result in self.results["methods"].items():
            cer = result["cer"]
            improvement = (baseline_cer - cer) / baseline_cer * 100
            method_cers.append((method_name, cer, improvement))
            
            status = "✅" if improvement > 0 else "⚠️"
            print(f"{status} {method_name:30} CER: {cer:5.1%}  (Δ {improvement:+5.1f}%)")
        
        # Inference Speed Comparison
        print("\n⏱️  Inference Speed Comparison")
        print("-" * 60)
        
        for method_name, result in sorted(
            self.results["methods"].items(),
            key=lambda x: x[1]["time_per_image"]
        ):
            time_per_img = result["time_per_image"]
            throughput = result["throughput"]
            print(f"   {method_name:30} {time_per_img:6.2f}s/img  ({throughput:5.2f} img/s)")
        
        # Accuracy vs Speed Trade-off
        print("\n📈 Accuracy vs Speed Trade-off")
        print("-" * 60)
        
        for method_name, result in self.results["methods"].items():
            cer = result["cer"]
            time_per_img = result["time_per_image"]
            improvement = (baseline_cer - cer) / baseline_cer * 100
            
            efficiency = improvement / (time_per_img + 0.1)  # Avoid division by zero
            
            print(f"   {method_name:30}")
            print(f"      Accuracy gain: {improvement:+6.1f}% | Time cost: {time_per_img:6.2f}s | Efficiency: {efficiency:5.1f}")
        
        # Recommendations
        print("\n🎯 Recommendations")
        print("-" * 60)
        
        best_cer_method = min(self.results["methods"].items(), 
                             key=lambda x: x[1]["cer"])
        best_speed_method = min(self.results["methods"].items(),
                               key=lambda x: x[1]["time_per_image"])
        
        best_cer_name, best_cer_result = best_cer_method
        best_speed_name, best_speed_result = best_speed_method
        
        print(f"\n🥇 BEST ACCURACY: {best_cer_name}")
        print(f"   CER: {best_cer_result['cer']:.1%}")
        print(f"   Time: {best_cer_result['time_per_image']:.2f}s per image")
        
        print(f"\n⚡ BEST SPEED: {best_speed_name}")
        print(f"   CER: {best_speed_result['cer']:.1%}")
        print(f"   Time: {best_speed_result['time_per_image']:.2f}s per image")
        
        # Find best balanced approach
        balanced_scores = []
        for method_name, result in self.results["methods"].items():
            cer_improvement = (baseline_cer - result["cer"]) / baseline_cer
            time_normalized = result["time_per_image"] / 100  # Normalize
            balance_score = cer_improvement - (time_normalized * 0.5)
            balanced_scores.append((method_name, balance_score, result))
        
        best_balanced = max(balanced_scores, key=lambda x: x[1])
        
        print(f"\n⚖️  BEST BALANCED: {best_balanced[0]}")
        print(f"   CER: {best_balanced[2]['cer']:.1%}")
        print(f"   Time: {best_balanced[2]['time_per_image']:.2f}s per image")
        print(f"   Balance Score: {best_balanced[1]:.3f}")
        
        # Production recommendations
        print(f"\n💼 PRODUCTION RECOMMENDATION")
        if best_cer_result["cer"] < baseline_cer * 0.75:  # 25% improvement
            print(f"   ✅ Use {best_cer_name} - Significant accuracy gain ({best_cer_result['cer']:.1%} CER)")
        else:
            print(f"   ⚙️  Use {best_balanced[0]} - Best accuracy/speed balance")
    
    def save_results(self, filename: str = "phase2_validation_results.json"):
        """Save benchmark results"""
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✅ Results saved to {filename}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 2 Optimization Benchmark Suite")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of test samples")
    parser.add_argument("--output", type=str, default="phase2_validation_results.json", help="Output file")
    
    args = parser.parse_args()
    
    # Run benchmark
    suite = OCRBenchmarkSuite()
    results = suite.run_full_benchmark(num_samples=args.num_samples)
    
    # Save results
    suite.save_results(args.output)
    
    print("\n" + "="*80)
    print("✅ BENCHMARK COMPLETE")
    print("="*80)
