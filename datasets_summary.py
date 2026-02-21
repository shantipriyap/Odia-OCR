#!/usr/bin/env python3
"""
Summary of available Odia OCR datasets for training
"""

datasets_summary = {
    "1. OdiaGenAIOCR/Odia-lipi-ocr-data": {
        "source": "HuggingFace Hub",
        "type": "OCR - Printed & Handwritten Text",
        "status": "✅ ACTIVE",
        "samples": "64+ samples",
        "features": ["image", "text"],
        "url": "https://huggingface.co/datasets/OdiaGenAIOCR/Odia-lipi-ocr-data",
        "notes": "Currently used in training",
        "access": "Public"
    },
    "2. tell2jyoti/odia-handwritten-ocr": {
        "source": "HuggingFace Hub",
        "type": "OCR - Handwritten Text",
        "status": "✅ ACTIVE",
        "samples": "Unknown (to explore)",
        "features": ["image", "text"],
        "url": "https://huggingface.co/datasets/tell2jyoti/odia-handwritten-ocr",
        "notes": "Handwritten Odia script recognition",
        "access": "Public"
    },
    "3. IIIT LOCR Odia Dataset": {
        "source": "IIIT Language Technologies Lab",
        "type": "OCR - Indic Language (Odia)",
        "status": "⭕ REQUIRES REGISTRATION",
        "samples": "Variable (check documentation)",
        "features": ["image", "text"],
        "url": "https://ilocr.iiit.ac.in/dataset/21/",
        "notes": "Part of IIIT Indic Language OCR Collection",
        "access": "Registration required via form",
        "registration": "https://ilocr.iiit.ac.in/dataset/21/"
    },
    "4. nayat/odia_text_classification": {
        "source": "HuggingFace Hub",
        "type": "Text Classification (not OCR)",
        "status": "⭕ NOT SUITABLE",
        "samples": "Unknown",
        "features": ["text", "label"],
        "url": "https://huggingface.co/datasets/nayat/odia_text_classification",
        "notes": "No image data, text only",
        "access": "Public"
    },
    "5. nayat/odia_squad": {
        "source": "HuggingFace Hub",
        "type": "QA (not OCR)",
        "status": "⭕ NOT SUITABLE",
        "samples": "Unknown",
        "features": ["question", "context", "answer"],
        "url": "https://huggingface.co/datasets/nayat/odia_squad",
        "notes": "Question-Answering dataset, no image data",
        "access": "Public"
    }
}

if __name__ == "__main__":
    print("\n" + "="*80)
    print("📊 ODIA OCR DATASETS SUMMARY")
    print("="*80 + "\n")
    
    suitable_for_ocr = []
    
    for name, info in datasets_summary.items():
        print(f"{name}")
        print(f"  Status: {info['status']}")
        print(f"  Source: {info['source']}")
        print(f"  Type: {info['type']}")
        print(f"  Samples: {info['samples']}")
        print(f"  Access: {info['access']}")
        if "notes" in info:
            print(f"  Notes: {info['notes']}")
        print(f"  URL: {info['url']}")
        print()
        
        if "OCR" in info['type'] and "✅" in info['status']:
            suitable_for_ocr.append(name)
    
    print("="*80)
    print("✅ SUITABLE FOR ODIA OCR TRAINING:\n")
    
    for ds in suitable_for_ocr:
        print(f"  • {ds}")
    
    print("\n" + "="*80)
    print("🚀 RECOMMENDED TRAINING STRATEGY")
    print("="*80 + "\n")
    
    print("Phase 1 (Quick - 30 min):")
    print("  1. Update training_ocr_qwen.py")
    print("  2. Combine OdiaGenAIOCR + tell2jyoti datasets")
    print("  3. Run 500 steps (vs 100 currently)")
    print("  4. Expected CER: 100% → 20-40%\n")
    
    print("Phase 2 (IIIT Dataset - 1-2 days):")
    print("  1. Register at: https://ilocr.iiit.ac.in/dataset/21/")
    print("  2. Download IIIT Odia OCR data")
    print("  3. Add to training pipeline")
    print("  4. Train 1000+ steps")
    print("  5. Expected CER: 20-40% → 10-25%\n")
    
    print("Phase 3 (Data Augmentation):")
    print("  1. Apply PIL augmentation (rotation, brightness, contrast)")
    print("  2. Increase training samples via synthesis")
    print("  3. Run 2000+ steps")
    print("  4. Expected CER: 10-25% → 5-15%\n")
    
    print("="*80)
    print("📝 NEXT STEPS:")
    print("="*80 + "\n")
    
    print("1. ✅ DONE: Updated training_ocr_qwen.py to support multiple datasets")
    print("2. ✅ DONE: Added tell2jyoti handwritten dataset support")
    print("3. ⏳ TODO: Test combined training (64 + N samples from tell2jyoti)")
    print("4. ⏳ TODO: Register & integrate IIIT dataset")
    print("5. ⏳ TODO: Implement data augmentation pipeline\n")
    
    print("="*80)
