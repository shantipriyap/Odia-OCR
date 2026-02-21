#!/usr/bin/env python3
"""
Discover available Odia OCR datasets on HuggingFace Hub
"""

from huggingface_hub import list_datasets
import json

def discover_odia_datasets():
    """Search for Odia-related OCR datasets"""
    
    print("\n" + "="*70)
    print("🔍 SEARCHING FOR ODIA OCR DATASETS ON HUGGINGFACE HUB")
    print("="*70 + "\n")
    
    # Search keywords
    search_terms = [
        "odia ocr",
        "odia text",
        "odia language",
        "oriya ocr",
        "oriya language",
        "odia script",
        "odia handwriting",
        "odia document",
        "odia character"
    ]
    
    found_datasets = {}
    
    for term in search_terms:
        try:
            print(f"🔎 Searching: '{term}'...")
            datasets = list_datasets(search=term, full=False, limit=5)
            
            for ds in datasets:
                if ds.id not in found_datasets:
                    found_datasets[ds.id] = {
                        "name": ds.id,
                        "description": ds.description or "No description",
                        "search_terms": [term]
                    }
                else:
                    found_datasets[ds.id]["search_terms"].append(term)
        except Exception as e:
            print(f"   ❌ Error searching '{term}': {e}")
    
    return found_datasets

def main():
    """Main discovery function"""
    
    # Known Odia datasets to include
    known_datasets = {
        "OdiaGenAIOCR/Odia-lipi-ocr-data": {
            "type": "OCR Text Recognition",
            "current": True,
            "description": "Odia handwritten and printed text dataset for OCR",
            "size": "64 samples (current)",
            "features": ["image", "text"],
        },
        "OdiaGenAIOCR/Odia-lipi-ocr-data-v2": {
            "type": "OCR Text Recognition",
            "current": False,
            "description": "Extended version with more samples",
            "size": "Unknown",
            "features": ["image", "text"],
        },
        "nayat/odia_text_classification": {
            "type": "Text Classification",
            "current": False,
            "description": "Odia text classification dataset",
            "size": "Unknown",
            "features": ["text", "label"],
        },
        "nayat/odia_squad": {
            "type": "QA Dataset",
            "current": False,
            "description": "Odia SQuAD (Question Answering)",
            "size": "Unknown",
            "features": ["question", "context", "answer"],
        },
        "indicnlp/indic_languages": {
            "type": "Multilingual",
            "current": False,
            "description": "Includes Odia as part of Indic languages dataset",
            "size": "Unknown",
            "features": ["text", "language"],
        },
    }
    
    print("\n📚 KNOWN ODIA DATASETS:\n")
    
    current_count = 0
    for dataset_id, info in known_datasets.items():
        marker = "✅" if info.get("current") else "⭕"
        print(f"{marker} {dataset_id}")
        print(f"   Type: {info['type']}")
        print(f"   Size: {info['size']}")
        print(f"   Features: {', '.join(info['features'])}")
        print(f"   Description: {info['description']}")
        if info.get("current"):
            current_count += 1
        print()
    
    print("-" * 70)
    print(f"📊 SUMMARY: {current_count} currently used, {len(known_datasets) - current_count} available")
    
    # Provide combining strategy
    print("\n" + "="*70)
    print("💡 DATA COMBINING STRATEGY")
    print("="*70 + "\n")
    
    strategies = [
        {
            "name": "Simple Concat (Recommended for Phase 1)",
            "pro": "Quick, no preprocessing needed",
            "con": "Different data distributions",
            "code": """
# Load multiple datasets
datasets = []
dataset1 = load_dataset("OdiaGenAIOCR/Odia-lipi-ocr-data")
datasets.append(dataset1["train"])

# Combine
combined = concatenate_datasets(datasets)
            """
        },
        {
            "name": "Balanced Mix (Phase 2)",
            "pro": "Prevents dataset bias",
            "con": "Requires sampling strategy",
            "code": """
# Sample equally from each source
from datasets import concatenate_datasets
dataset1 = load_dataset("OdiaGenAIOCR/Odia-lipi-ocr-data")["train"]
# Add more datasets...
combined = concatenate_datasets([d.select(range(min(100, len(d)))) for d in datasets])
            """
        },
        {
            "name": "Stratified Mix (Phase 3)",
            "pro": "Optimal data distribution",
            "con": "Requires analysis",
            "code": """
# Analyze + combine based on characteristics
# - OCR: 50%
# - Text Recognition: 30%
# - Synthetic/Augmented: 20%
            """
        }
    ]
    
    for i, strategy in enumerate(strategies, 1):
        print(f"{i}. {strategy['name']}")
        print(f"   ✅ Pro: {strategy['pro']}")
        print(f"   ❌ Con: {strategy['con']}")
        print()
    
    # Save results
    config = {
        "known_datasets": known_datasets,
        "discovered_datasets": {},
        "recommendation": "Start with OdiaGenAIOCR/Odia-lipi-ocr-data + v2 if available",
        "strategies": strategies
    }
    
    with open("odia_datasets_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Configuration saved to: odia_datasets_config.json")
    
    return known_datasets

if __name__ == "__main__":
    datasets = main()
