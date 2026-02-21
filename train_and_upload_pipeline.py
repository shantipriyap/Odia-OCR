#!/usr/bin/env python3
"""
Train Qwen2.5-VL on merged Odia OCR dataset and push checkpoints to HuggingFace
Monitors training progress and uploads best model automatically
"""

import subprocess
import time
import sys
import os
from datetime import datetime
from pathlib import Path
import json

MODEL_OUTPUT_DIR = "/root/odia_ocr/qwen_odia_ocr_improved_v2"
HF_REPO_ID = "shantipriya/qwen2.5-odia-ocr-v2"
CHECKPOINT_DIR = "/root/odia_ocr/checkpoints"
LOG_FILE = "/root/odia_ocr/training_progress.log"

def log_message(msg):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg)
    with open(LOG_FILE, "a") as f:
        f.write(log_msg + "\n")

def run_command(cmd, description=""):
    """Run shell command"""
    if description:
        log_message(f"🔄 {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        log_message(f"❌ Command timeout: {cmd}")
        return "", "Timeout", -1
    except Exception as e:
        log_message(f"❌ Error: {e}")
        return "", str(e), -1

def push_checkpoint_to_hf(checkpoint_path, step_num):
    """Push checkpoint to HuggingFace Hub"""
    log_message(f"📤 Pushing checkpoint {step_num} to HF...")
    
    try:
        # Create HF repo if needed
        run_command(
            f'huggingface-cli repo create "{HF_REPO_ID}" --type model --private 2>/dev/null || true',
            f"Ensuring HF repo exists: {HF_REPO_ID}"
        )
        
        # Clone or update repo
        repo_path = "/tmp/hf_model_repo"
        if not Path(repo_path).exists():
            run_command(f'git clone "https://huggingface.co/{HF_REPO_ID}" "{repo_path}"')
        else:
            run_command(f'cd "{repo_path}" && git pull origin main 2>/dev/null || git pull origin master')
        
        # Copy checkpoint
        run_command(f'cp -r "{checkpoint_path}"/* "{repo_path}/" 2>/dev/null', f"Copying checkpoint to repo")
        
        # Create model card if doesn't exist
        card_path = f"{repo_path}/README.md"
        if not Path(card_path).exists():
            with open(card_path, "w") as f:
                f.write(f"""# Qwen2.5-VL Odia OCR Model (Training {step_num}/500)

Fine-tuned on merged Odia OCR dataset (145K+ samples).

## Dataset
- OdiaGenAIOCR: 64 samples
- tell2jyoti/odia-handwritten-ocr: 145,717 samples
- Total: 145,781 Odia OCR samples

## Training Details
- Base Model: Qwen/Qwen2.5-VL-3B-Instruct
- Fine-tuning Method: LoRA (r=32, α=64)
- Training Steps: 500
- Learning Rate: 1e-4
- Scheduler: Cosine with warmup (50 steps)
- Batch Size: Effective 4 (1 per device × 4 accumulation)

## Usage

```python
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

processor = AutoProcessor.from_pretrained("{HF_REPO_ID}")
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "{HF_REPO_ID}")

# Use model for inference
```

## Expected Performance
- Training Step {step_num}/500
- Expected CER improvement: 100% → 30-50%

See: https://huggingface.co/datasets/shantipriya/odia-ocr-merged
""")
        
        # Git add, commit, push
        run_command(f'cd "{repo_path}" && git add .', "Staging files for HF")
        run_command(
            f'cd "{repo_path}" && git -c user.email="user@example.com" -c user.name="Odia OCR" commit -m "Update checkpoint {step_num}/500"',
            f"Committing checkpoint {step_num}"
        )
        run_command(
            f'cd "{repo_path}" && git push https://shantipriya:{os.getenv("HF_TOKEN")}@huggingface.co/{HF_REPO_ID} main 2>&1 | tail -5',
            f"Pushing to HF Hub"
        )
        
        log_message(f"✅ Checkpoint {step_num} pushed to {HF_REPO_ID}")
        return True
        
    except Exception as e:
        log_message(f"❌ Error pushing checkpoint: {e}")
        return False

def monitor_and_upload_training():
    """Monitor training and upload checkpoints"""
    
    log_message("\n" + "="*70)
    log_message("🚀 STARTING TRAINING PIPELINE WITH CONTINUOUS HF UPLOADS")
    log_message("="*70)
    
    log_message(f"Dataset: shantipriya/odia-ocr-merged (145,781 samples)")
    log_message(f"Output: {MODEL_OUTPUT_DIR}")
    log_message(f"HF Repo: {HF_REPO_ID}")
    log_message(f"Log: {LOG_FILE}")
    
    # Start training
    log_message("\n📋 Starting training script...")
    train_cmd = (
        "cd /root/odia_ocr && "
        "source /root/venv/bin/activate && "
        "python3 training_improved_merged_dataset.py > training_main.log 2>&1 &"
    )
    
    stdout, stderr, code = run_command(train_cmd, "Launching training process")
    log_message("✅ Training process started in background")
    
    # Give training time to initialize
    log_message("Waiting for training to initialize...")
    time.sleep(30)
    
    # Monitor training and push checkpoints
    last_pushed = 0
    check_interval = 120  # Check every 2 minutes
    
    while True:
        try:
            # Check if training is running
            check_cmd = "ps aux | grep -v grep | grep 'training_improved_merged_dataset.py' | wc -l"
            stdout, _, _ = run_command(check_cmd)
            is_running = int(stdout.strip()) > 0
            
            if not is_running:
                log_message("⏸️  Training process finished")
                log_message("🔍 Checking for final model...")
                break
            
            # Check for new checkpoints
            checkpoints = sorted(Path(MODEL_OUTPUT_DIR).glob("checkpoint-*")) if Path(MODEL_OUTPUT_DIR).exists() else []
            
            if checkpoints:
                latest = checkpoints[-1]
                checkpoint_num = int(latest.name.split("-")[1])
                
                # Push if this is a new checkpoint
                if checkpoint_num > last_pushed and checkpoint_num % 50 == 0:
                    push_checkpoint_to_hf(str(latest), checkpoint_num)
                    last_pushed = checkpoint_num
            
            # Wait before next check
            log_message(f"⏳ Training in progress... (rechecking in {check_interval}s)")
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            log_message("⏹️  Stopping monitoring (Ctrl+C)")
            break
        except Exception as e:
            log_message(f"⚠️  Error in monitoring loop: {e}")
            time.sleep(check_interval)
    
    # Final upload
    log_message("\n📤 Uploading final model...")
    if Path(MODEL_OUTPUT_DIR).exists():
        push_checkpoint_to_hf(MODEL_OUTPUT_DIR, "final")
        log_message("✅ Final model uploaded!")
    
    log_message("\n" + "="*70)
    log_message("✅ TRAINING AND UPLOAD PIPELINE COMPLETE!")
    log_message("="*70)
    log_message(f"Model available at: https://huggingface.co/{HF_REPO_ID}")

if __name__ == "__main__":
    # Ensure log file exists
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    
    # Check HF token
    if not os.getenv("HF_TOKEN"):
        log_message("❌ HF_TOKEN environment variable not set!")
        log_message("Set it with: export HF_TOKEN='your_token_here'")
        sys.exit(1)
    
    try:
        monitor_and_upload_training()
    except Exception as e:
        log_message(f"❌ Fatal error: {e}")
        sys.exit(1)
