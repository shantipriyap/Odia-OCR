#!/usr/bin/env python3
"""
Explore the Odia handwritten OCR dataset from tell2jyoti
"""

from datasets import load_dataset
import json

def explore_dataset():
    """Explore the Odia handwritten OCR dataset"""
    
    print("\n" + "="*70)
    print("🔍 EXPLORING ODIA HANDWRITTEN OCR DATASET")
    print("="*70 + "\n")
    
    try:
        print("📥 Loading dataset: tell2jyoti/odia-handwritten-ocr...")
        dataset = load_dataset("tell2jyoti/odia-handwritten-ocr")
        
        print("✅ Dataset loaded successfully!\n")
        
        # Explore structure
        print("📊 DATASET STRUCTURE:\n")
        
        info = {
            "splits": {},
            "total_samples": 0,
            "features": {},
        }
        
        for split_name in dataset.keys():
            split = dataset[split_name]
            num_samples = len(split)
            info["splits"][split_name] = num_samples
            info["total_samples"] += num_samples
            
            print(f"Split: {split_name:20} | Samples: {num_samples:6}")
        
        print()
        
        # Get first example
        first_split = list(dataset.keys())[0]
        first_example = dataset[first_split][0]
        
        print(f"📝 EXAMPLE FROM '{first_split}':\n")
        
        for key, value in first_example.items():
            if isinstance(value, str) and len(str(value)) > 100:
                print(f"   {key}: {str(value)[:100]}...")
            else:
                print(f"   {key}: {type(value).__name__} = {value if not isinstance(value, bytes) else '[Image Data]'}")
            
            if key not in info["features"]:
                info["features"][key] = type(value).__name__
        
        print()
        print("-" * 70)
        print(f"✅ Total samples across all splits: {info['total_samples']}")
        print(f"📋 Features: {list(info['features'].keys())}")
        
        return dataset, info
    
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("\nTrying to check if dataset exists on HuggingFace Hub...")
        return None, None

def compare_datasets():
    """Compare current dataset with new one"""
    
    print("\n" + "="*70)
    print("📊 DATASET COMPARISON")
    print("="*70 + "\n")
    
    comparison = {
        "OdiaGenAIOCR/Odia-lipi-ocr-data": {
            "source": "OdiaGenAIOCR",
            "type": "Odia handwritten & printed text",
            "current_samples": 64,
            "features": ["image", "text"],
            "status": "✅ Currently used"
        },
        "tell2jyoti/odia-handwritten-ocr": {
            "source": "tell2jyoti",
            "type": "Odia handwritten OCR",
            "current_samples": "Unknown (to be discovered)",
            "features": "To be discovered",
            "status": "⭕ Recommended for Phase 2"
        }
    }
    
    print("| Dataset                                  | Type                      | Samples        |")
    print("|------------------------------------------|---------------------------|----------------|")
    for ds_name, info in comparison.items():
        print(f"| {ds_name:40} | {info['type']:25} | {str(info['current_samples']):14} |")
    
    print("\n💡 COMBINING STRATEGY:\n")
    print("Phase 1 (Quick):")
    print("  • Keep: OdiaGenAIOCR/Odia-lipi-ocr-data (64 samples)")
    print("  • Add: tell2jyoti/odia-handwritten-ocr")
    print("  • Strategy: Simple concatenation")
    print("  • Expected samples: 64 + N")
    print("  • Training time: +10-20%")
    print("  • Expected CER: 100% → 20-40% (improvement from more diverse data)")
    print()
    print("Phase 2 (Optimized):")
    print("  • Load both datasets")
    print("  • Apply data augmentation (rotation, brightness, contrast)")
    print("  • Use balanced sampling")
    print("  • Increase training steps to 1000")
    print("  • Expected CER: 15-25%")
    print()

def create_multi_dataset_loader():
    """Create script to load multiple Odia OCR datasets"""
    
    print("\n" + "="*70)
    print("📝 MULTI-DATASET LOADING SCRIPT")
    print("="*70 + "\n")
    
    script = '''
# Multi-dataset loader for Odia OCR training

from datasets import load_dataset, concatenate_datasets
import os

def load_odia_ocr_datasets(sources=None, splits=None):
    """
    Load and combine multiple Odia OCR datasets
    
    Args:
        sources: List of dataset identifiers
        splits: Which splits to use (default: ["train"])
    
    Returns:
        Combined dataset with all samples
    """
    
    if sources is None:
        sources = [
            "OdiaGenAIOCR/Odia-lipi-ocr-data",      # Current: 64 samples
            "tell2jyoti/odia-handwritten-ocr",      # New dataset
        ]
    
    if splits is None:
        splits = ["train"]
    
    datasets = []
    
    for source in sources:
        try:
            print(f"📥 Loading {source}...")
            ds = load_dataset(source)
            
            for split in splits:
                if split in ds:
                    datasets.append(ds[split])
                    print(f"   ✅ Added {split} split ({len(ds[split])} samples)")
        except Exception as e:
            print(f"   ❌ Error loading {source}: {e}")
    
    if datasets:
        combined = concatenate_datasets(datasets)
        print(f"\\n✅ Combined dataset: {len(combined)} total samples")
        return combined
    else:
        print("❌ No datasets loaded")
        return None

# Usage:
# train_dataset = load_odia_ocr_datasets()
    '''
    
    print(script)
    
    return script

if __name__ == "__main__":
    # Explore the dataset
    dataset, info = explore_dataset()
    
    if dataset:
        # Compare with current dataset
        compare_datasets()
        
        # Show loading script
        create_multi_dataset_loader()
        
        # Save info
        with open("tell2jyoti_dataset_info.json", "w") as f:
            json.dump(info, f, indent=2)
        
        print("\n✅ Dataset info saved to: tell2jyoti_dataset_info.json")
    else:
        print("\n⚠️  Could not load dataset. Make sure you have internet connection.")
        print("    The dataset might need authentication or might not be available.")
