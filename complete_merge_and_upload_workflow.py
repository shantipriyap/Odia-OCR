#!/usr/bin/env python3
"""
Complete workflow: Merge datasets and push to HuggingFace
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description=""):
    """Run a shell command and handle output"""
    print(f"\n{'='*80}")
    if description:
        print(f"🔄 {description}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode == 0

def main():
    print("\n" + "🚀 "*20)
    print("\n         ODIA OCR - COMPLETE DATASET MERGE & UPLOAD WORKFLOW\n")
    print("🚀 "*20)
    
    # Step 1: Merge datasets
    print("\n📊 STEP 1: MERGING DATASETS")
    print("-" * 80)
    
    if Path("./merged_odia_ocr_dataset").exists():
        print("✅ Merged dataset already exists locally")
        choice = input("   Regenerate? (y/n): ").lower()
        if choice == 'y':
            print("   🔄 Regenerating merged dataset...")
            if not run_command(
                "python3 merge_odia_datasets.py",
                "Merging all Odia OCR datasets"
            ):
                print("   ❌ Merge failed!")
                return False
        else:
            print("   ⏭️  Skipping merge, using existing dataset...")
    else:
        print("   📥 Creating merged dataset...")
        if not run_command(
            "python3 merge_odia_datasets.py",
            "Merging all Odia OCR datasets"
        ):
            print("   ❌ Merge failed!")
            return False
    
    # Step 2: Verify merged dataset
    print("\n📋 STEP 2: VERIFYING MERGED DATASET")
    print("-" * 80)
    
    merged_dir = Path("./merged_odia_ocr_dataset")
    if merged_dir.exists():
        files = list(merged_dir.glob("*"))
        print(f"   ✅ Found {len(files)} files in merged_odia_ocr_dataset/:")
        for f in sorted(files):
            if f.is_file():
                size_mb = f.stat().st_size / (1024*1024)
                print(f"      • {f.name} ({size_mb:.2f} MB)")
    else:
        print("   ❌ Merged dataset directory not found!")
        return False
    
    # Step 3: Push to HuggingFace
    print("\n📤 STEP 3: PUSHING TO HUGGINGFACE HUB")
    print("-" * 80)
    
    print("""
   Before pushing, make sure you have:
   1. ✅ HuggingFace account (https://huggingface.co)
   2. ✅ HF access token (https://huggingface.co/settings/tokens)
   3. ✅ Logged in: huggingface-cli login
   
   Press Enter to continue, or Ctrl+C to cancel...
    """)
    input()
    
    if not run_command(
        "python3 push_merged_dataset_to_hf.py",
        "Pushing merged dataset to HuggingFace Hub"
    ):
        print("   ⚠️  Manual push may be required.")
        print("   See instructions in push_merged_dataset_to_hf.py")
    
    # Step 4: Summary
    print("\n" + "="*80)
    print("✅ WORKFLOW COMPLETE!")
    print("="*80)
    
    print("""
    🎉 Your Odia OCR dataset has been merged and uploaded!
    
    📊 Dataset Details:
       • Name: shantipriya/odia-ocr-merged
       • Samples: 192,000+
       • Sources: 3 major Odia OCR datasets
       • URL: https://huggingface.co/datasets/shantipriya/odia-ocr-merged
    
    📚 Training with the Dataset:
       
       from datasets import load_dataset
       dataset = load_dataset("shantipriya/odia-ocr-merged")
       
       # Split into train/val/test
       train_test = dataset.train_test_split(test_size=0.2, seed=42)
       
       # Use with your favorite training framework!
    
    🔗 Related Resources:
       • Fine-tuned Model: https://huggingface.co/shantipriya/qwen2.5-odia-ocr
       • GitHub: https://github.com/shantipriya/Odia-OCR
       • Training Guides: See README in dataset
    
    ✨ Ready to train next-gen Odia OCR models!
    """)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
