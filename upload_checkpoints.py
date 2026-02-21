#!/usr/bin/env python3
"""
Upload Odia OCR checkpoints to HuggingFace Hub
Automatically uploads new checkpoints and creates model card
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime
import json

REMOTE_HOST = "135.181.8.206"
REMOTE_CKPT_DIR = "/root/odia_ocr/qwen_odia_ocr_improved_v2"
LOCAL_OUTPUT_DIR = "./qwen_odia_ocr_improved_v2"
HF_REPO = "shantipriya/qwen2.5-odia-ocr-v2"
HF_TOKEN = os.environ.get("HF_TOKEN", "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")

print("""
╔════════════════════════════════════════════════════════════════╗
║          🚀 ODIA OCR CHECKPOINT UPLOADER 🚀                   ║
║                                                                ║
║  Upload checkpoints from remote training to HuggingFace       ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
""")

def ssh_cmd(cmd):
    """Run SSH command"""
    try:
        result = subprocess.run(
            ["ssh", f"root@{REMOTE_HOST}", cmd],
            capture_output=True,
            text=True,
            timeout=20
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception as e:
        print(f"SSH error: {e}")
        return None

def get_remote_checkpoints():
    """Get list of checkpoints on remote"""
    cmd = f"ls -d {REMOTE_CKPT_DIR}/checkpoint-* 2>/dev/null | sort -V"
    output = ssh_cmd(cmd)
    
    checkpoints = []
    if output:
        for line in output.split('\n'):
            if line:
                checkpoint_name = line.split('/')[-1]
                checkpoints.append(checkpoint_name)
    return checkpoints

def download_checkpoint(checkpoint_name):
    """Download checkpoint from remote"""
    remote_path = f"{REMOTE_HOST}:{REMOTE_CKPT_DIR}/{checkpoint_name}"
    local_path = f"{LOCAL_OUTPUT_DIR}/{checkpoint_name}"
    
    # Create local dir if needed
    Path(LOCAL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    if os.path.exists(local_path):
        print(f"✅ {checkpoint_name} already downloaded")
        return True
    
    print(f"\n📥 Downloading {checkpoint_name}...", end=" ", flush=True)
    try:
        cmd = f"scp -qr root@{remote_path} {local_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, timeout=300)
        
        if result.returncode == 0:
            size_mb = sum(f.stat().st_size for f in Path(local_path).rglob('*')) / (1024*1024)
            print(f"✅ ({size_mb:.1f} MB)")
            return True
        else:
            print(f"❌ Failed")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def upload_checkpoint_to_hf(checkpoint_name):
    """Upload checkpoint to HuggingFace Hub"""
    local_path = f"{LOCAL_OUTPUT_DIR}/{checkpoint_name}"
    
    if not os.path.exists(local_path):
        print(f"   ❌ Local path not found: {local_path}")
        return False
    
    print(f"🚀 Uploading {checkpoint_name} to HF...", end=" ", flush=True)
    
    try:
        # Use huggingface_hub for uploading
        from huggingface_hub import upload_folder
        
        step_num = int(checkpoint_name.replace("checkpoint-", ""))
        
        # Upload entire checkpoint directory
        upload_folder(
            folder_path=local_path,
            repo_id=HF_REPO,
            path_in_repo=f"checkpoints/{checkpoint_name}",
            repo_type="model",
            token=HF_TOKEN,
            private=False,
            commit_message=f"Add checkpoint {checkpoint_name} (step {step_num}/500)"
        )
        
        print(f"✅")
        return True
        
    except ImportError:
        print(f"⚠️ (needs huggingface_hub)")
        print(f"   Install: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ ({str(e)[:50]})")
        return False

def create_model_card():
    """Create README for model on HF"""
    model_card_path = f"{LOCAL_OUTPUT_DIR}/README.md"
    
    readme_content = """---
license: mit
tags:
  - odia
  - ocr
  - vision-language
  - qwen2.5-vl
  - lora
  - fine-tuning
language:
  - or
datasets:
  - shantipriya/odia-ocr-merged
---

# Qwen2.5-VL Fine-tuned for Odia OCR

This is a fine-tuned version of [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) trained on the [shantipriya/odia-ocr-merged](https://huggingface.co/datasets/shantipriya/odia-ocr-merged) dataset.

## 📊 Training Details

- **Base Model**: Qwen/Qwen2.5-VL-3B-Instruct
- **Dataset**: shantipriya/odia-ocr-merged (145,781 samples)
- **Method**: LoRA fine-tuning
  - Rank: 32
  - Alpha: 64
  - Target Modules: q_proj, v_proj
- **Training Config**:
  - Steps: 500
  - Learning Rate: 1e-4
  - Scheduler: Cosine with warmup (50 steps)
  - Batch Size: 4 (1 per device × 4 gradient accumulation)
- **Training Data Split**: 80/10/10 (train/val/test)

## 🎯 Performance

**Expected Improvement**:
- Baseline CER: 100%
- Phase 1 Target: 30-50% CER
- Expected Improvement: 10-40x

## 📝 Dataset Details

The merged dataset combines three Odia OCR sources:

1. **OdiaGenAIOCR/Odia-lipi-ocr-data**: 64 word-level samples
2. **tell2jyoti/odia-handwritten-ocr**: 145,717 character-level samples (32×32px, 47 OHCS characters)
3. **darknight054/indic-mozhi-ocr**: 10,000+ printed word samples

**Total**: 145,781 unique Odia text-image pairs

## 🚀 Usage

```python
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import requests

# Load model and processor
model_id = "shantipriya/qwen2.5-odia-ocr-v2"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype="auto",
    trust_remote_code=True,
    device_map="auto"
)

# Load image
url = "https://example.com/odia_text.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Process and generate
inputs = processor(image, text="Recognize Odia text:", return_tensors="pt")
output_ids = model.generate(**inputs)
text = processor.decode(output_ids[0])
print(text)
```

## 📂 Checkpoints

This model contains checkpoints saved at every 50 training steps:
- `checkpoints/checkpoint-50` (10%)
- `checkpoints/checkpoint-100` (20%)
- `checkpoints/checkpoint-150` (30%)
- ... and more

See the model repository for all available checkpoints.

## 🔗 Related Resources

- **Merged Dataset**: https://huggingface.co/datasets/shantipriya/odia-ocr-merged
- **Base Model**: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
- **PEFT**: https://huggingface.co/docs/peft/

## 📄 License

MIT

## 👏 Acknowledgments

- **Dataset Sources**: OdiaGenAIOCR, tell2jyoti, darknight054
- **Framework**: Hugging Face Transformers, PEFT
- **Base Model**: Qwen Team

---

*Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(model_card_path, 'w') as f:
        f.write(readme_content)
    
    print(f"📄 Model card created: {model_card_path}")

def main():
    """Main upload process"""
    
    # Get checkpoints
    print("📋 Scanning remote checkpoints...")
    remote_ckpts = get_remote_checkpoints()
    
    if not remote_ckpts:
        print("❌ No checkpoints found on remote")
        return
    
    print(f"✅ Found {len(remote_ckpts)} checkpoints\n")
    
    # Download each checkpoint
    downloaded = []
    for ckpt in remote_ckpts:
        if download_checkpoint(ckpt):
            downloaded.append(ckpt)
    
    print(f"\n✅ Downloaded {len(downloaded)} checkpoints locally")
    
    # Create model card
    print("\n" + "="*70)
    create_model_card()
    
    # Upload to HF
    print("\n" + "="*70)
    print("🚀 UPLOADING TO HUGGINGFACE")
    print("="*70)
    
    uploaded = 0
    for ckpt in downloaded:
        if upload_checkpoint_to_hf(ckpt):
            uploaded += 1
    
    # Summary
    print("\n" + "="*70)
    print("📊 UPLOAD SUMMARY")
    print("="*70)
    print(f"✅ Downloaded: {len(downloaded)}/{len(remote_ckpts)}")
    print(f"✅ Uploaded: {uploaded}/{len(downloaded)}")
    print(f"\n🔗 Model URL: https://huggingface.co/{HF_REPO}")
    print("\n💡 Next: Monitor training completion then upload final model")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Upload cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
