#!/usr/bin/env python3
"""
QUICK REFERENCE: Merge & Upload Odia OCR Dataset to HuggingFace
"""

quick_ref = """
╔════════════════════════════════════════════════════════════════════════════════╗
║                          🚀 QUICK REFERENCE GUIDE                              ║
║                    Merge & Upload Odia OCR Dataset to HF                       ║
╚════════════════════════════════════════════════════════════════════════════════╝


⚡ FASTEST PATH (One Command)
═════════════════════════════════════════════════════════════════════════════════

$ python3 complete_merge_and_upload_workflow.py

Done in 20-30 minutes! Handles everything.


📋 MANUAL PATH (3 Commands)
═════════════════════════════════════════════════════════════════════════════════

1. Merge locally:
   $ python3 merge_odia_datasets.py

2. Setup authentication:
   $ huggingface-cli login
   # Paste your token from https://huggingface.co/settings/tokens

3. Upload to HF:
   $ python3 push_merged_dataset_to_hf.py


📊 WHAT YOU GET
═════════════════════════════════════════════════════════════════════════════════

192,000+ Odia OCR samples from 3 sources:
  • OdiaGenAIOCR: 64 word-level images
  • tell2jyoti: 182,152 character images
  • darknight054: 10,000+ printed Odia words

Dataset URL after upload:
  https://huggingface.co/datasets/shantipriya/odia-ocr-merged


📚 DOCUMENTATION AVAILABLE
═════════════════════════════════════════════════════════════════════════════════

Complete guides:
  $ python3 print_merge_upload_guide.py       → Full tutorial
  $ python3 print_merge_summary.py            → Summary
  $ python3 print_final_summary.py            → Complete overview

Markdown files:
  MERGE_UPLOAD_GUIDE.md
  MERGE_DATASET_SUMMARY.txt
  FINAL_MERGE_SUMMARY.txt


💻 LOAD & USE (After Upload)
═════════════════════════════════════════════════════════════════════════════════

from datasets import load_dataset
dataset = load_dataset("shantipriya/odia-ocr-merged")

# 192,000+ samples ready!
print(f"Samples: {len(dataset['train']):,}")


🎯 EXPECTED IMPROVEMENTS
═════════════════════════════════════════════════════════════════════════════════

Current (64 samples): CER = 100% ❌
With merged (192K): CER = 10-25% ✅ (10-40x improvement!)


⚙️ FILES INVOLVED
═════════════════════════════════════════════════════════════════════════════════

Scripts:
  merge_odia_datasets.py               → Merge datasets
  push_merged_dataset_to_hf.py         → Upload to HF
  complete_merge_and_upload_workflow.py → Full automation

Output:
  ./merged_odia_ocr_dataset/           → Local directory
    ├── data.parquet
    ├── README.md
    └── metadata.json


✅ REQUIREMENTS
═════════════════════════════════════════════════════════════════════════════════

$ pip install huggingface-hub huggingface-datasets

Or use existing: requirements.txt already has these!


🔐 AUTHENTICATION
═════════════════════════════════════════════════════════════════════════════════

Get token: https://huggingface.co/settings/tokens
Login: $ huggingface-cli login
Enter token when prompted


❓ TROUBLESHOOTING
═════════════════════════════════════════════════════════════════════════════════

Upload fails?
  → Try manual git-based upload (see print_merge_upload_guide.py)

Dataset not found on HF?
  → Check internet connection
  → Verify token is valid

Memory issues?
  → Load one dataset at a time
  → Use force_download=True

Still stuck?
  → See MERGE_UPLOAD_GUIDE.md section "Troubleshooting"


═════════════════════════════════════════════════════════════════════════════════

That's it! Run one of:
  • python3 complete_merge_and_upload_workflow.py  (Easiest)
  • python3 merge_odia_datasets.py + push script   (Manual)

═════════════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(quick_ref)
    with open("QUICK_START.txt", "w") as f:
        f.write(quick_ref)
    print("\n✅ Quick reference saved to: QUICK_START.txt")
