#!/usr/bin/env python3
"""
Simple script to push merged Odia OCR dataset to HuggingFace Hub
"""

import os
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi, HfFolder, list_repo_files
import sys

def main():
    print("\n" + "="*80)
    print("🚀 PUSHING MERGED ODIA OCR DATASET TO HUGGINGFACE HUB")
    print("="*80 + "\n")
    
    # Step 1: Check dataset
    dataset_dir = Path("./merged_odia_ocr_dataset")
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return False
    
    parquet_file = dataset_dir / "data.parquet"
    if not parquet_file.exists():
        print(f"❌ Parquet file not found: {parquet_file}")
        return False
    
    print(f"✅ Found dataset: {parquet_file}")
    print(f"   File size: {parquet_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Step 2: Load dataset
    print("\n📥 Loading dataset...")
    try:
        dataset = load_dataset("parquet", data_files=str(parquet_file))
        num_samples = len(dataset["train"])
        print(f"   ✅ Loaded: {num_samples:,} samples")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Step 3: Get token
    print("\n🔐 Checking HuggingFace authentication...")
    token = HfFolder.get_token()
    if not token:
        print("   ❌ No HuggingFace token found!")
        print("\n   To authenticate:")
        print("   1. Get a token from: https://huggingface.co/settings/tokens")
        print("   2. Run: huggingface-cli login")
        print("   3. Paste your token when prompted")
        return False
    
    print(f"   ✅ Token found")
    
    # Step 4: Push to hub
    repo_id = "shantipriya/odia-ocr-merged"
    print(f"\n📤 Pushing to HuggingFace Hub...")
    print(f"   Repository: {repo_id}")
    print(f"   Samples: {num_samples:,}")
    print(f"   This may take 5-15 minutes...\n")
    
    try:
        dataset["train"].push_to_hub(
            repo_id=repo_id,
            split="train",
            private=False,
            token=token
        )
        print(f"\n   ✅ ✅ ✅ Dataset pushed successfully!")
        
    except Exception as e:
        print(f"\n   ❌ Error: {e}")
        return False
    
    # Step 5: Success
    print("\n" + "="*80)
    print("✅ ✅ ✅ DATASET UPLOAD COMPLETE!")
    print("="*80)
    print(f"\n📊 DATASET DETAILS:")
    print(f"   Name: odia-ocr-merged")
    print(f"   Repository: https://huggingface.co/datasets/{repo_id}")
    print(f"   Samples: {num_samples:,}")
    print(f"   Status: 🌐 Public")
    print(f"\n🎉 Your dataset is live on HuggingFace Hub!")
    print(f"\n📖 TO USE YOUR DATASET:")
    print(f"   from datasets import load_dataset")
    print(f"   dataset = load_dataset('{repo_id}')")
    print(f"\n📚 NEXT STEPS:")
    print(f"   1. Visit: https://huggingface.co/datasets/{repo_id}")
    print(f"   2. Edit README and add comprehensive training guide")
    print(f"   3. Share with the Odia community!")
    print(f"   4. Start training: python3 training_ocr_qwen.py")
    print()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
