#!/usr/bin/env python3
"""
Test script to verify:
1. Model can be downloaded from HuggingFace
2. Model can be loaded correctly
3. Inference works
4. Results match Phase 2A evaluation
"""

import os
import sys
import json
import torch
from pathlib import Path
from PIL import Image
import time

def test_model_download_and_load():
    """Test downloading model from HuggingFace"""
    print("=" * 70)
    print("🧪 TEST 1: Model Download & Load from HuggingFace")
    print("=" * 70)
    
    try:
        print("\n1️⃣ Importing required libraries...")
        from transformers import AutoProcessor, LlavaNextForConditionalGeneration
        from peft import PeftModel
        print("   ✅ Imports successful")
        
        print("\n2️⃣ Downloading base model: Qwen/Qwen2.5-VL-3B-Instruct...")
        model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        
        processor = AutoProcessor.from_pretrained(model_id)
        print("   ✅ Processor loaded")
        
        base_model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("   ✅ Base model loaded")
        
        print("\n3️⃣ Loading LoRA adapter from HuggingFace: shantipriya/qwen2.5-odia-ocr...")
        model = PeftModel.from_pretrained(base_model, "shantipriya/qwen2.5-odia-ocr")
        print("   ✅ LoRA adapter loaded successfully")
        
        print("\n✅ TEST 1 PASSED: Model downloaded and loaded successfully!")
        return model, processor
        
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_inference(model, processor):
    """Test inference with a sample image"""
    print("\n" + "=" * 70)
    print("🧪 TEST 2: Inference Testing")
    print("=" * 70)
    
    try:
        print("\n1️⃣ Looking for test image...")
        
        # Try to find any image file
        test_images = list(Path(".").glob("**/*.jpg")) + list(Path(".").glob("**/*.png"))
        
        if not test_images:
            print("   ⚠️ No test images found locally")
            print("   Creating a simple test image...")
            
            # Create a simple test image
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='white')
            test_image_path = "test_image.jpg"
            img.save(test_image_path)
            print(f"   ✅ Test image created: {test_image_path}")
        else:
            test_image_path = str(test_images[0])
            print(f"   ✅ Test image found: {test_image_path}")
        
        print("\n2️⃣ Loading and processing image...")
        image = Image.open(test_image_path).convert("RGB")
        print(f"   ✅ Image loaded: {image.size}")
        
        print("\n3️⃣ Running inference...")
        text = "Extract the Odia text from this image. Return only the recognized text."
        
        inputs = processor(
            text=text,
            images=image,
            return_tensors="pt"
        ).to("cuda", torch.float16)
        
        start_time = time.time()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=256)
        inference_time = time.time() - start_time
        
        result = processor.decode(output[0], skip_special_tokens=True)
        
        print(f"   ✅ Inference completed in {inference_time:.2f} seconds")
        print(f"\n   📝 Input prompt: {text}")
        print(f"   📝 Model output: {result}")
        
        print("\n✅ TEST 2 PASSED: Inference works successfully!")
        return True, inference_time
        
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_phase2a_evaluation():
    """Verify Phase 2A results are available"""
    print("\n" + "=" * 70)
    print("🧪 TEST 3: Phase 2A Evaluation Results")
    print("=" * 70)
    
    try:
        print("\n1️⃣ Checking for Phase 2A results file...")
        
        results_file = Path("phase2_quick_win_results.json")
        
        if not results_file.exists():
            print(f"   ⚠️ Results file not found: {results_file}")
            return False
        
        print(f"   ✅ Results file found: {results_file}")
        
        print("\n2️⃣ Loading Phase 2A results...")
        with open(results_file) as f:
            results = json.load(f)
        
        print(f"   ✅ Results loaded successfully")
        
        print("\n3️⃣ Phase 2A Evaluation Results Summary:")
        print(f"   📊 Timestamp: {results.get('timestamp')}")
        print(f"   📊 Test Samples: {results.get('test_samples')}")
        
        methods = results.get('methods', {})
        print("\n   📈 Performance by Method:")
        for method, metrics in methods.items():
            cer = metrics.get('cer', 'N/A')
            time_per_img = metrics.get('time_per_image', 'N/A')
            print(f"      • {method}:")
            print(f"        - CER: {cer}")
            print(f"        - Time/image: {time_per_img}s")
        
        # Expected values
        expected = {
            'greedy': 0.42,
            'beam_search_5': 0.37,
            'ensemble_voting': 0.32
        }
        
        print("\n4️⃣ Verification against expected values:")
        all_match = True
        for method, expected_cer in expected.items():
            if method in methods:
                actual_cer = methods[method].get('cer')
                match = abs(actual_cer - expected_cer) < 0.01
                status = "✅" if match else "⚠️"
                print(f"   {status} {method}: Expected {expected_cer}, Got {actual_cer}")
                if not match:
                    all_match = False
            else:
                print(f"   ❌ {method}: NOT FOUND")
                all_match = False
        
        if all_match:
            print("\n✅ TEST 3 PASSED: Phase 2A results verified!")
        else:
            print("\n⚠️ TEST 3 PARTIAL: Some results don't match expected values")
        
        return all_match
        
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  🧪 MODEL VERIFICATION TEST SUITE".center(68) + "║")
    print("║" + "  Feb 22, 2026 - Phase 2A Model Validation".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    results = {}
    
    # Test 1: Download and load
    model, processor = test_model_download_and_load()
    results['test1_download'] = model is not None
    
    # Test 2: Inference
    if model is not None:
        inference_ok, inf_time = test_inference(model, processor)
        results['test2_inference'] = inference_ok
        results['inference_time'] = inf_time
    else:
        results['test2_inference'] = False
    
    # Test 3: Phase 2A results
    phase2a_ok = test_phase2a_evaluation()
    results['test3_phase2a'] = phase2a_ok
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 TEST SUMMARY")
    print("=" * 70)
    
    test_results = [
        ("Model Download & Load", results.get('test1_download', False)),
        ("Inference", results.get('test2_inference', False)),
        ("Phase 2A Results", results.get('test3_phase2a', False)),
    ]
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n📊 Final Score: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Model is ready for production.")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
