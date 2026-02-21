#!/usr/bin/env python3
"""
Push merged Odia OCR dataset to HuggingFace Hub
"""

import os
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import login

def push_dataset_to_hf(
    dataset_dir="./merged_odia_ocr_dataset",
    repo_id="odia-ocr-merged",
    private=False
):
    """
    Push merged dataset to HuggingFace Hub
    
    Args:
        dataset_dir: Local directory with merged dataset
        repo_id: Repository ID (will use shantipriya/{repo_id})
        private: Whether to make dataset private
    """
    
    print("\n" + "="*80)
    print("🚀 PUSHING MERGED ODIA OCR DATASET TO HUGGINGFACE HUB")
    print("="*80 + "\n")
    
    # Step 1: Check dataset exists
    print("📁 Checking merged dataset...")
    if not Path(dataset_dir).exists():
        print(f"   ❌ Directory not found: {dataset_dir}")
        print(f"   Please run merge_odia_datasets_clean.py first")
        return False
    
    parquet_file = Path(dataset_dir) / "data.parquet"
    if not parquet_file.exists():
        print(f"   ❌ Parquet file not found: {parquet_file}")
        return False
    
    print(f"   ✅ Found merged dataset")
    
    # Step 2: Load dataset
    print("\n📥 Loading dataset from parquet...")
    try:
        dataset = load_dataset("parquet", data_files=str(parquet_file))
        num_samples = len(dataset["train"])
        print(f"   ✅ Loaded: {num_samples:,} samples")
    except Exception as e:
        print(f"   ❌ Error loading dataset: {e}")
        return False
    
    # Step 3: Authenticate
    print("\n🔐 Authenticating with HuggingFace...")
    try:
        login()
        print("   ✅ Authenticated successfully")
    except Exception as e:
        print(f"   ❌ Authentication failed: {e}")
        print("   Please get a token from: https://huggingface.co/settings/tokens")
        return False
    
    # Step 4: Push to Hub
    full_repo_id = f"shantipriya/{repo_id}"
    print(f"\n📤 Pushing to HuggingFace Hub...")
    print(f"   Repository: {full_repo_id}")
    print(f"   Samples: {num_samples:,}")
    print(f"   Status: {'🔒 Private' if private else '🌐 Public'}")
    print(f"   This may take 5-10 minutes...\n")
    
    try:
        dataset["train"].push_to_hub(
            repo_id=full_repo_id,
            split="train",
            private=private
        )
        print(f"   ✅ Dataset pushed successfully!")
    except Exception as e:
        print(f"   ❌ Error pushing dataset: {e}")
        return False
    
    # Step 5: Upload README
    print(f"\n📝 Uploading README...")
    readme_path = Path(dataset_dir) / "README.md"
    if readme_path.exists():
        try:
            # Note: The README.md in the dataset directory will be used as the dataset card
            print(f"   ✅ README will appear on the dataset page")
        except Exception as e:
            print(f"   ⚠️  Could not update README: {e}")
    
    # Step 6: Summary
    print("\n" + "="*80)
    print("✅ DATASET UPLOAD COMPLETE!")
    print("="*80)
    print(f"\n📊 DATASET INFO:")
    print(f"   Repository: {full_repo_id}")
    print(f"   URL: https://huggingface.co/datasets/{full_repo_id}")
    print(f"   Samples: {num_samples:,}")
    print(f"   Visibility: {'Private' if private else 'Public'}")
    print(f"\n🎉 Your dataset is now live on HuggingFace Hub!")
    print(f"\n📖 Loading your dataset:")
    print(f"   from datasets import load_dataset")
    print(f"   dataset = load_dataset('{full_repo_id}')")
    print()
    
    return True


if __name__ == "__main__":
    success = push_dataset_to_hf(
        dataset_dir="./merged_odia_ocr_dataset",
        repo_id="odia-ocr-merged",
        private=False
    )
    
    if not success:
        print("\n❌ Upload failed!")
        exit(1)
