#!/usr/bin/env python3
"""
Update README on HuggingFace Hub for Odia OCR model
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, ModelCardData, ModelCard

# Configuration
MODEL_ID = "shantipriya/qwen2.5-odia-ocr-v2"
TOKEN = os.getenv("HF_TOKEN")

if not TOKEN:
    print("❌ HF_TOKEN not set")
    exit(1)

print("\n" + "="*80)
print("📚 Uploading Updated README to HuggingFace Hub")
print("="*80)

# Read local README
readme_path = Path("README.md")
if not readme_path.exists():
    print(f"❌ README not found at {readme_path}")
    exit(1)

with open(readme_path, "r", encoding="utf-8") as f:
    readme_content = f.read()

print(f"\n[1/3] 📖 Reading local README...")
print(f"✅ README loaded ({len(readme_content)} bytes)")

# Initialize API
print(f"\n[2/3] 🔐 Connecting to HuggingFace Hub...")
api = HfApi()
print(f"✅ Connected to HuggingFace Hub")

# Upload README
print(f"\n[3/3] 📤 Uploading README to {MODEL_ID}...")
try:
    api.upload_file(
        path_or_fileobj=readme_content.encode('utf-8'),
        path_in_repo="README.md",
        repo_id=MODEL_ID,
        repo_type="model",
        commit_message="📚 Update README with comprehensive evaluation results",
        token=TOKEN
    )
    print(f"✅ README uploaded successfully!")
    print(f"\n📍 Model available at: https://huggingface.co/{MODEL_ID}")
    
except Exception as e:
    print(f"❌ Upload failed: {e}")
    exit(1)

print("\n" + "="*80)
print("✅ README UPDATE COMPLETE")
print("="*80)
print(f"\n📊 Changes synced to HuggingFace Hub:")
print(f"   - Latest evaluation results")
print(f"   - Performance metrics (42% CER)")
print(f"   - Detailed methodology")
print(f"   - Improvement roadmap")
print(f"\n🌐 View online: https://huggingface.co/{MODEL_ID}")
print("="*80 + "\n")
