#!/usr/bin/env python3
"""
Push merged Odia OCR dataset to HuggingFace Hub (Non-interactive)
"""

import os
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi, HfFolder

def push_dataset_to_hf_noninteractive(
    dataset_dir="./merged_odia_ocr_dataset",
    repo_id="odia-ocr-merged",
    private=False
):
    """
    Push merged dataset to HuggingFace Hub without interactive login
    Requires HF_TOKEN environment variable to be set
    
    Args:
        dataset_dir: Local directory with merged dataset
        repo_id: Repository ID (will use shantipriya/{repo_id})
        private: Whether to make dataset private
    """
    
    print("\n" + "="*80)
    print("🚀 PUSHING MERGED ODIA OCR DATASET TO HUGGINGFACE HUB")
    print("="*80 + "\n")
    
    # Step 1: Check token
    token = os.getenv("HF_TOKEN") or HfFolder.get_token()
    if not token:
        print("❌ ERROR: HF_TOKEN not found!")
        print("\nPlease set your HuggingFace token:")
        print("  export HF_TOKEN='your_token_here'")
        print("\nOr create a token at: https://huggingface.co/settings/tokens")
        return False
    
    print("✅ HuggingFace token found")
    
    # Step 2: Check dataset exists
    print("\n📁 Checking merged dataset...")
    if not Path(dataset_dir).exists():
        print(f"   ❌ Directory not found: {dataset_dir}")
        print(f"   Please run merge_odia_datasets_clean.py first")
        return False
    
    parquet_file = Path(dataset_dir) / "data.parquet"
    if not parquet_file.exists():
        print(f"   ❌ Parquet file not found: {parquet_file}")
        return False
    
    print(f"   ✅ Found merged dataset")
    
    # Step 3: Load dataset
    print("\n📥 Loading dataset from parquet...")
    try:
        dataset = load_dataset("parquet", data_files=str(parquet_file))
        num_samples = len(dataset["train"])
        print(f"   ✅ Loaded: {num_samples:,} samples")
    except Exception as e:
        print(f"   ❌ Error loading dataset: {e}")
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
            private=private,
            token=token
        )
        print(f"   ✅ Dataset pushed successfully!")
    except Exception as e:
        print(f"   ❌ Error pushing dataset: {e}")
        return False
    
    # Step 5: Summary
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
    success = push_dataset_to_hf_noninteractive(
        dataset_dir="./merged_odia_ocr_dataset",
        repo_id="odia-ocr-merged",
        private=False
    )
    
    if not success:
        print("\n❌ Upload failed!")
        exit(1)
