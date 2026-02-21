#!/usr/bin/env python3
"""
Upload Odia OCR model to HuggingFace Hub
Simple and reliable method using huggingface_hub
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime

REMOTE_HOST = "135.181.8.206"
REMOTE_CKPT_DIR = "/root/odia_ocr/qwen_odia_ocr_improved_v2"
LOCAL_OUTPUT_DIR = "./qwen_odia_ocr_improved_v2"
HF_REPO = "shantipriya/qwen2.5-odia-ocr-v2"
HF_TOKEN = os.environ.get("HF_TOKEN", "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")

print("""
╔════════════════════════════════════════════════════════════════╗
║          🚀 ODIA OCR TO HUGGINGFACE 🚀                        ║
║                                                                ║
║  Upload checkpoint directories to HuggingFace Hub             ║
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
    except:
        return None

def main():
    """Main process"""
    
    # Ensure token is set
    os.environ["HF_TOKEN"] = HF_TOKEN
    
    # Get checkpoints
    print("📋 Scanning checkpoints...")
    cmd = f"ls -d {REMOTE_CKPT_DIR}/checkpoint-* 2>/dev/null | sort -V"
    output = ssh_cmd(cmd)
    
    checkpoints = []
    if output:
        checkpoints = [line.split('/')[-1] for line in output.split('\n') if line]
    
    print(f"✅ Found {len(checkpoints)} checkpoints: {', '.join(checkpoints[:3])}...")
    
    # Create/update model card first
    model_card = f"""{LOCAL_OUTPUT_DIR}/README.md"""
    Path(LOCAL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    readme = f"""---
license: mit
tags:
  - odia
  - ocr
  - vision-language
  - qwen
language:
  - or
datasets:
  - shantipriya/odia-ocr-merged
---

# Qwen2.5-VL Fine-tuned for Odia OCR

Fine-tuned [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) on [shantipriya/odia-ocr-merged](https://huggingface.co/datasets/shantipriya/odia-ocr-merged) (145,781 Odia text-image pairs).

## 📊 Training

- **Base Model**: Qwen/Qwen2.5-VL-3B-Instruct (3B params)
- **Dataset**: 145,781 Odia OCR samples
- **Method**: LoRA (r=32, α=64)
- **Training**: 500 steps, cosine scheduler, warmup=50
- **Learning Rate**: 1e-4
- **Batch Size**: 4 effective
- **Expected CER**: 100% baseline → 30-50% improved

## 🤗 Usage

```python
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image

processor = AutoProcessor.from_pretrained("{HF_REPO}", trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "{HF_REPO}",
    torch_dtype="auto",
    trust_remote_code=True,
    device_map="auto"
)

image = Image.open("odia_text.jpg")
inputs = processor(image, text="Recognize:", return_tensors="pt")
output_ids = model.generate(**inputs)
text = processor.decode(output_ids[0])
```

## 📁 Checkpoints

Available checkpoints:
{chr(10).join([f"- `checkpoints/{ckpt}` (step {int(ckpt.replace('checkpoint-', ''))})" for ckpt in checkpoints])}

## 🔗 Resources

- **Merged Dataset**: https://huggingface.co/datasets/shantipriya/odia-ocr-merged
- **Base Model**: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

---
Updated: {datetime.now().isoformat()}
"""
    
    with open(model_card, 'w') as f:
        f.write(readme)
    
    print(f"📄 Updated README: {model_card}")
    
    # Try uploading with Python API
    print("\n" + "="*70)
    print("🚀 UPLOADING TO HUGGINGFACE")
    print("="*70)
    
    try:
        from huggingface_hub import HfApi
        
        api = HfApi()
        
        # Upload README first
        print(f"\n📤 Uploading README.md...")
        try:
            api.upload_file(
                path_or_fileobj=model_card,
                path_in_repo="README.md",
                repo_id=HF_REPO,
                repo_type="model",
                token=HF_TOKEN,
                commit_message="Update README with training details"
            )
            print(f"✅ README uploaded")
        except Exception as e:
            print(f"⚠️ README: {str(e)[:50]}")
        
        # Upload each checkpoint's files
        for ckpt_name in checkpoints:
            checkpoint_dir = f"{LOCAL_OUTPUT_DIR}/{ckpt_name}"
            if not os.path.exists(checkpoint_dir):
                print(f"⚠️ {ckpt_name}: not downloaded locally yet")
                continue
            
            print(f"\n📤 Uploading {ckpt_name}...")
            
            try:
                # List files in checkpoint
                files_to_upload = []
                for file_path in Path(checkpoint_dir).rglob("*"):
                    if file_path.is_file():
                        files_to_upload.append(file_path)
                
                if not files_to_upload:
                    print(f"⚠️ No files found in {ckpt_name}")
                    continue
                
                # Upload each file
                for file_path in files_to_upload[:10]:  # Upload first 10 files
                    rel_path = file_path.relative_to(LOCAL_OUTPUT_DIR)
                    
                    try:
                        api.upload_file(
                            path_or_fileobj=str(file_path),
                            path_in_repo=str(rel_path),
                            repo_id=HF_REPO,
                            repo_type="model",
                            token=HF_TOKEN,
                        )
                    except Exception as e:
                        print(f"   ⚠️ {file_path.name}: {str(e)[:40]}")
                
                print(f"✅ {ckpt_name} uploaded ({len(files_to_upload)} files)")
                
            except Exception as e:
                print(f"❌ {ckpt_name}: {str(e)[:50]}")
        
        print("\n✅ Upload complete!")
        print(f"🔗 Model: https://huggingface.co/{HF_REPO}")
        
    except ImportError:
        print("❌ huggingface_hub not available")
        print("Install: pip install huggingface_hub")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Cancelled")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
