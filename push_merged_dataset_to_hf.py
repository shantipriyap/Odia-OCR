#!/usr/bin/env python3
"""
Push merged Odia OCR dataset to HuggingFace Hub
"""

from datasets import Dataset, DatasetDict, load_dataset
from pathlib import Path
import json
import os
from huggingface_hub import login, create_repo, Repository

def push_merged_dataset_to_hf(
    dataset_name="odia-ocr-merged",
    private=False,
    use_auth_token=True
):
    """
    Push merged dataset to HuggingFace Hub
    
    Args:
        dataset_name: Name for the dataset on HF (will be: shantipriya/{dataset_name})
        private: Whether to make dataset private
        use_auth_token: Whether to use HF auth token
    """
    
    print("\n" + "="*80)
    print("🚀 PUSHING MERGED ODIA OCR DATASET TO HUGGINGFACE HUB")
    print("="*80 + "\n")
    
    # Step 1: Authenticate with HuggingFace
    if use_auth_token:
        print("🔐 Authenticating with HuggingFace...")
        try:
            login()
            print("   ✅ Authenticated successfully\n")
        except Exception as e:
            print(f"   ⚠️  Note: You can set HF_TOKEN environment variable")
            print(f"   Error: {e}\n")
    
    # Step 2: Load merged dataset
    print("📥 Loading merged dataset from local...")
    merged_dir = "./merged_odia_ocr_dataset"
    
    if not Path(merged_dir).exists():
        print(f"   ❌ Directory not found: {merged_dir}")
        print(f"   Run: python3 merge_odia_datasets.py")
        return False
    
    try:
        dataset = load_dataset("parquet", data_files=f"{merged_dir}/data.parquet")
        print(f"   ✅ Loaded dataset with {len(dataset['train']):,} samples\n")
    except Exception as e:
        print(f"   ❌ Error loading dataset: {e}")
        return False
    
    # Step 3: Load metadata
    print("📋 Loading dataset metadata...")
    try:
        with open(f"{merged_dir}/metadata.json", "r") as f:
            metadata = json.load(f)
        print(f"   ✅ Metadata loaded\n")
    except Exception as e:
        print(f"   ⚠️  Could not find metadata: {e}")
        metadata = {}
    
    # Step 4: Load README
    print("📖 Loading README...")
    readme_path = f"{merged_dir}/README.md"
    if Path(readme_path).exists():
        with open(readme_path, "r") as f:
            readme_content = f.read()
        print(f"   ✅ README loaded ({len(readme_content)} chars)\n")
    else:
        print(f"   ⚠️  README not found, will create default\n")
        readme_content = f"""# {metadata.get('name', f'{dataset_name} Dataset')}

{metadata.get('description', 'Odia OCR merged dataset')}

## Dataset Size
- **Total Samples**: {len(dataset['train']):,}

## Loading the Dataset

```python
from datasets import load_dataset
dataset = load_dataset("shantipriya/{dataset_name}")
```

See the full README at: https://huggingface.co/datasets/shantipriya/{dataset_name}
"""
    
    # Step 5: Create dataset repository
    hf_dataset_name = f"shantipriya/{dataset_name}"
    print(f"🏗️  Creating repository on HuggingFace Hub...")
    print(f"   Dataset name: {hf_dataset_name}\n")
    
    try:
        create_repo(
            repo_id=hf_dataset_name,
            repo_type="dataset",
            private=private,
            exist_ok=True
        )
        print(f"   ✅ Repository ready\n")
    except Exception as e:
        print(f"   ⚠️  Repository creation note: {e}\n")
    
    # Step 6: Push to Hub
    print("📤 Pushing dataset to HuggingFace Hub...")
    print(f"   This may take a few minutes...\n")
    
    try:
        # Push the dataset
        dataset["train"].push_to_hub(
            repo_id=hf_dataset_name,
            split="train",
            private=private
        )
        print(f"   ✅ Dataset pushed successfully!\n")
    except Exception as e:
        print(f"   ❌ Error pushing dataset: {e}")
        return False
    
    # Step 7: Upload README to dataset card
    print("📝 Creating dataset card...")
    try:
        # Create a simple dataset_info.json
        dataset_info = {
            "name": metadata.get('name', f'{dataset_name}'),
            "description": metadata.get('description', 'Odia OCR merged dataset'),
            "sources": metadata.get('sources', {}),
            "total_samples": len(dataset["train"]),
        }
        
        with open(f"{merged_dir}/dataset_info.json", "w") as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"   ✅ Dataset card created\n")
    except Exception as e:
        print(f"   ⚠️  Could not create dataset card: {e}\n")
    
    # Step 8: Summary
    print("="*80)
    print("✅ DATASET UPLOAD COMPLETE!")
    print("="*80)
    print(f"\n📊 DATASET DETAILS:")
    print(f"   Name: {hf_dataset_name}")
    print(f"   Samples: {len(dataset['train']):,}")
    print(f"   URL: https://huggingface.co/datasets/{hf_dataset_name}")
    print(f"   Status: {'🔒 Private' if private else '🌐 Public'}")
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Visit: https://huggingface.co/datasets/{hf_dataset_name}")
    print(f"   2. Edit dataset card and add the comprehensive README")
    print(f"   3. Share with community!")
    print()
    
    return True

def manual_push_instructions():
    """Print manual push instructions"""
    
    print("\n" + "="*80)
    print("📋 MANUAL PUSH INSTRUCTIONS (Alternative)")
    print("="*80 + "\n")
    
    print("""
If automatic push fails, follow these manual steps:

1. Install/Update huggingface_hub:
   pip install --upgrade huggingface-hub

2. Login to HuggingFace:
   huggingface-cli login
   # Then paste your access token from https://huggingface.co/settings/tokens

3. Create dataset repository:
   huggingface-cli repo create odia-ocr-merged --type dataset

4. Clone the repository:
   git clone https://huggingface.co/datasets/shantipriya/odia-ocr-merged
   cd odia-ocr-merged

5. Copy merged dataset files:
   cp ../merged_odia_ocr_dataset/data.parquet ./
   cp ../merged_odia_ocr_dataset/README.md ./
   cp ../merged_odia_ocr_dataset/metadata.json ./

6. Create dataset_infojson:
   cat > dataset_info.json << 'EOF'
   {
     "name": "Odia OCR - Merged Multi-Source Dataset",
     "description": "Combined Odia OCR dataset from multiple sources for training",
     "features": ["image", "text"]
   }
   EOF

7. Commit and push:
   git add .
   git commit -m "📚 Add merged Odia OCR dataset"
   git push

8. Update dataset card:
   # Edit README.md on the HuggingFace web interface
   """)

if __name__ == "__main__":
    import sys
    
    # Check if datasets are already merged
    if not Path("./merged_odia_ocr_dataset").exists():
        print("❌ Merged dataset not found!")
        print("First, run: python3 merge_odia_datasets.py")
        sys.exit(1)
    
    # Push to HuggingFace
    success = push_merged_dataset_to_hf(
        dataset_name="odia-ocr-merged",
        private=False,
        use_auth_token=True
    )
    
    if not success:
        print("\n⚠️  Automatic push failed. See manual instructions below...\n")
        manual_push_instructions()
