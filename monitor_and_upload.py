#!/usr/bin/env python3
"""
Real-time monitoring of Odia OCR training
Tracks metrics, detects new checkpoints, and uploads to HuggingFace Hub
"""

import os
import re
import time
import subprocess
from datetime import datetime
from pathlib import Path
import json

# Configuration
REMOTE_HOST = "135.181.8.206"
REMOTE_USER = "root"
REMOTE_LOG = "/root/odia_ocr/training_main.log"
REMOTE_OUTPUT_DIR = "/root/odia_ocr/qwen_odia_ocr_improved_v2"
LOCAL_OUTPUT_DIR = "./qwen_odia_ocr_improved_v2"
HF_REPO = "shantipriya/qwen2.5-odia-ocr-v2"
HF_TOKEN = os.environ.get("HF_TOKEN")
MONITOR_INTERVAL = 30  # Check every 30 seconds

print("""
╔════════════════════════════════════════════════════════════════╗
║           📊 ODIA OCR TRAINING PERFORMANCE MONITOR 📊          ║
║                                                                ║
║  • Tracks training metrics (loss, progress, speed)             ║
║  • Detects new checkpoints automatically                       ║
║  • Uploads to HuggingFace Hub every 50 steps                   ║
║  • Generates real-time performance report                      ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# SSH UTILITIES
# ============================================================================

def ssh_command(cmd):
    """Run command on remote server via SSH"""
    try:
        result = subprocess.run(
            ["ssh", f"{REMOTE_USER}@{REMOTE_HOST}", cmd],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout if result.returncode == 0 else None
    except Exception as e:
        print(f"   ⚠️  SSH error: {e}")
        return None

# ============================================================================
# MONITORING FUNCTIONS
# ============================================================================

def get_log_tail(lines=50):
    """Fetch last N lines from training log"""
    cmd = f"tail -{lines} {REMOTE_LOG}"
    output = ssh_command(cmd)
    return output if output else ""

def parse_training_progress():
    """Extract training progress from log"""
    log = get_log_tail(20)
    
    # Look for progress line like: 32%|███▏      | 161/500 [01:24<02:56,  1.92it/s]
    pattern = r'(\d+)%\|.*?\|\s*(\d+)/(\d+).*?\[(.*?)<(.*?),\s*([\d.]+)it/s\]'
    match = re.search(pattern, log)
    
    if match:
        percent = int(match.group(1))
        current = int(match.group(2))
        total = int(match.group(3))
        elapsed = match.group(4)
        remaining = match.group(5)
        speed = float(match.group(6))
        
        return {
            "percent": percent,
            "current": current,
            "total": total,
            "elapsed": elapsed,
            "remaining": remaining,
            "speed": speed,
            "status": "running"
        }
    return None

def get_loss_from_log():
    """Extract loss values from log"""
    log = get_log_tail(50)
    
    # Look for loss patterns
    loss_pattern = r'loss["\']?\s*:\s*([\d.]+)'
    matches = re.findall(loss_pattern, log, re.IGNORECASE)
    
    if matches:
        return [float(m) for m in matches[-5:]]  # Last 5 loss values
    return []

def get_checkpoints():
    """List saved checkpoints on remote"""
    cmd = f"ls -d {REMOTE_OUTPUT_DIR}/checkpoint-* 2>/dev/null | sort -V"
    output = ssh_command(cmd)
    
    if output:
        return [os.path.basename(line.strip()) for line in output.strip().split('\n') if line.strip()]
    return []

def check_training_running():
    """Check if training process is still running"""
    cmd = "pgrep -f 'python3 training_simple_v6.py' | wc -l"
    output = ssh_command(cmd)
    if output:
        return int(output.strip()) > 0
    return False

# ============================================================================
# UPLOAD FUNCTIONS
# ============================================================================

def download_checkpoint(checkpoint_name):
    """Download checkpoint from remote to local"""
    remote_path = f"{REMOTE_OUTPUT_DIR}/{checkpoint_name}"
    local_path = f"{LOCAL_OUTPUT_DIR}/{checkpoint_name}"
    
    print(f"\n   📥 Downloading {checkpoint_name}...")
    try:
        cmd = f"scp -r {REMOTE_USER}@{REMOTE_HOST}:{remote_path} {local_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, timeout=300)
        
        if result.returncode == 0:
            print(f"   ✅ Downloaded to: {local_path}")
            return True
        else:
            print(f"   ❌ Download failed: {result.stderr.decode()}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def upload_to_hf(checkpoint_name):
    """Upload checkpoint to HuggingFace Hub"""
    local_path = f"{LOCAL_OUTPUT_DIR}/{checkpoint_name}"
    
    if not os.path.exists(local_path):
        print(f"   ❌ Local path not found: {local_path}")
        return False
    
    print(f"\n   🚀 Uploading {checkpoint_name} to HF Hub...")
    
    try:
        # Upload to HF using huggingface_hub
        from huggingface_hub import HfApi
        
        api = HfApi()
        step_num = checkpoint_name.replace("checkpoint-", "")
        
        # Upload directory
        api.upload_folder(
            folder_path=local_path,
            repo_id=HF_REPO,
            path_in_repo=f"checkpoints/{checkpoint_name}",
            repo_type="model",
            token=HF_TOKEN,
            private=False,
            commit_message=f"Add {checkpoint_name}: Step {step_num}/500"
        )
        
        print(f"   ✅ Uploaded to: https://huggingface.co/{HF_REPO}")
        return True
    except Exception as e:
        print(f"   ⚠️  HF upload error: {e}")
        print(f"   💡 Make sure: python -m pip install huggingface_hub")
        return False

# ============================================================================
# REPORTING
# ============================================================================

def print_status_report(progress, checkpoints, uploaded):
    """Print formatted status report"""
    if not progress:
        print("   ⏳ Waiting for training to start...")
        return
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*70}")
    print(f"  📊 TRAINING STATUS - {timestamp}")
    print(f"{'='*70}")
    print(f"  Progress:    {progress['current']:,}/{progress['total']} steps ({progress['percent']}%)")
    print(f"  Speed:       {progress['speed']:.2f} it/s")
    print(f"  Elapsed:     {progress['elapsed']}")
    print(f"  Remaining:   ~{progress['remaining']}")
    print(f"  Checkpoints: {len(checkpoints)} saved")
    print(f"  Uploaded:    {uploaded} to HF Hub")
    
    losses = get_loss_from_log()
    if losses:
        print(f"  Latest Loss: {losses[-1]:.4f}", end="")
        if len(losses) > 1 and losses[-2] > 0:
            improvement = ((losses[-2] - losses[-1]) / losses[-2]) * 100
            arrow = "📉" if improvement > 0 else "📈"
            print(f"  {arrow} ({improvement:+.1f}%)")
        else:
            print()
    
    print(f"{'='*70}\n")

# ============================================================================
# MAIN MONITORING LOOP
# ============================================================================

def main():
    """Main monitoring loop"""
    uploaded_checkpoints = set()
    last_status_report = None
    
    print(f"🟢 Starting monitor (checks every {MONITOR_INTERVAL}s)...\n")
    
    try:
        while check_training_running():
            # Get current state
            progress = parse_training_progress()
            checkpoints = get_checkpoints()
            new_checkpoints = set(checkpoints) - uploaded_checkpoints
            
            # Print status
            if progress and (progress != last_status_report):
                print_status_report(progress, checkpoints, len(uploaded_checkpoints))
                last_status_report = progress
            
            # Upload new checkpoints
            for checkpoint in sorted(new_checkpoints):
                try:
                    print(f"\n🔔 New checkpoint detected: {checkpoint}")
                    
                    # Download and upload
                    if download_checkpoint(checkpoint):
                        if upload_to_hf(checkpoint):
                            uploaded_checkpoints.add(checkpoint)
                            print(f"   ✅ Successfully uploaded {checkpoint}")
                except Exception as e:
                    print(f"   ⚠️  Error uploading {checkpoint}: {e}")
            
            # Wait before next check
            time.sleep(MONITOR_INTERVAL)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped by user")
    
    # Final report
    print(f"\n{'='*70}")
    print(f"📊 FINAL SUMMARY")
    print(f"{'='*70}")
    progress = parse_training_progress()
    if progress:
        print(f"✅ Training Progress: {progress['current']}/{progress['total']} steps ({progress['percent']}%)")
    print(f"✅ Checkpoints Uploaded: {len(uploaded_checkpoints)}")
    print(f"✅ Model Available at: https://huggingface.co/{HF_REPO}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
