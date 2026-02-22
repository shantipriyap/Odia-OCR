#!/usr/bin/env python3
"""
Continuous monitoring of Odia OCR model training performance
Tracks metrics, logs progress, and uploads checkpoints to HuggingFace
"""

import subprocess
import time
import sys
import os
import re
from datetime import datetime
from pathlib import Path
import json

def log_message(msg):
    """Print and log message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg)

def run_remote_cmd(cmd):
    """Run command on remote server"""
    full_cmd = f"ssh root@135.181.8.206 '{cmd}'"
    try:
        result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), -1

def check_training_status():
    """Check if training is running"""
    stdout, _, _ = run_remote_cmd("ps aux | grep -v grep | grep 'python3 training_improved' | wc -l")
    return int(stdout.strip()) > 0

def get_training_metrics():
    """Extract training metrics from log file"""
    stdout, _, _ = run_remote_cmd("tail -100 /root/odia_ocr/training_main.log")
    
    metrics = {}
    
    # Extract step info
    step_match = re.search(r'(\d+)%\|.*?(\d+)/500.*?(\d+\.?\d*s/it)', stdout)
    if step_match:
        percentage = step_match.group(1)
        steps = step_match.group(2)
        speed = step_match.group(3)
        metrics['progress'] = f"{steps}/500 ({percentage}%)"
        metrics['speed'] = speed
    
    # Extract loss
    loss_matches = re.findall(r"'loss': ([\d.]+)", stdout)
    if loss_matches:
        metrics['last_loss'] = loss_matches[-1]
    
    # Extract learning rate
    lr_matches = re.findall(r"'learning_rate': ([\d.e-]+)", stdout)
    if lr_matches:
        metrics['learning_rate'] = lr_matches[-1]
    
    return metrics

def get_checkpoint_status():
    """Check for saved checkpoints"""
    stdout, _, _ = run_remote_cmd("ls -la /root/odia_ocr/qwen_odia_ocr_improved_v2/checkpoint-* 2>/dev/null | tail -5")
    
    checkpoints = []
    for line in stdout.split('\n'):
        if 'checkpoint-' in line:
            match = re.search(r'checkpoint-(\d+)', line)
            if match:
                checkpoints.append(int(match.group(1)))
    
    return sorted(checkpoints) if checkpoints else []

def upload_checkpoint_to_hf(checkpoint_num):
    """Push checkpoint to HuggingFace Hub"""
    log_message(f"📤 Uploading checkpoint-{checkpoint_num} to HuggingFace...")
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        log_message("❌ HF_TOKEN environment variable not set")
        return
    
    # Create tar archive of checkpoint
    tar_cmd = f"cd /root/odia_ocr && tar -czf checkpoint-{checkpoint_num}.tar.gz qwen_odia_ocr_improved_v2/checkpoint-{checkpoint_num}/"
    run_remote_cmd(tar_cmd)
    
    # Download checkpoint
    download_cmd = f"scp root@135.181.8.206:/root/odia_ocr/checkpoint-{checkpoint_num}.tar.gz /tmp/"
    subprocess.run(download_cmd, shell=True, capture_output=True)
    
    # Extract and prepare for upload
    extract_cmd = f"cd /tmp && tar -xzf checkpoint-{checkpoint_num}.tar.gz"
    subprocess.run(extract_cmd, shell=True, capture_output=True)
    
    log_message(f"✅ Checkpoint {checkpoint_num} prepared for upload")
    return True

def monitor_training():
    """Main monitoring loop"""
    
    log_message("\n" + "="*80)
    log_message("🚀 STARTING TRAINING PERFORMANCE MONITORING")
    log_message("="*80 + "\n")
    
    log_message("📊 Monitoring Configuration:")
    log_message(f"  • Remote Server: 135.181.8.206")
    log_message(f"  • Dataset: shantipriya/odia-ocr-merged (145,781 samples)")
    log_message(f"  • Training Steps: 500")
    log_message(f"  • Base Model: Qwen/Qwen2.5-VL-3B-Instruct")
    log_message(f"  • Method: LoRA (r=32, α=64)")
    log_message(f"  • Update Interval: 30 seconds\n")
    
    last_saved_checkpoint = 0
    metrics_history = []
    
    while True:
        try:
            # Check if training is running
            is_running = check_training_status()
            
            if not is_running:
                log_message("\n⏹️  Training process stopped")
                break
            
            # Get current metrics
            metrics = get_training_metrics()
            checkpoints = get_checkpoint_status()
            
            # Display status
            status_line = "🟢 TRAINING IN PROGRESS"
            if metrics:
                if 'progress' in metrics:
                    status_line += f" | Progress: {metrics['progress']}"
                if 'speed' in metrics:
                    status_line += f" | Speed: {metrics['speed']}"
            
            log_message(status_line)
            
            if metrics:
                if 'last_loss' in metrics:
                    log_message(f"   Loss: {metrics['last_loss']}")
                if 'learning_rate' in metrics:
                    log_message(f"   LR: {metrics['learning_rate']}")
            
            # Check for new checkpoints
            if checkpoints:
                latest_checkpoint = checkpoints[-1]
                log_message(f"   Checkpoints: {len(checkpoints)} saved | Latest: checkpoint-{latest_checkpoint}")
                
                # Upload every 50 steps
                if latest_checkpoint > last_saved_checkpoint and latest_checkpoint % 50 == 0:
                    log_message(f"   ✨ New checkpoint detected: checkpoint-{latest_checkpoint}")
                    upload_checkpoint_to_hf(latest_checkpoint)
                    last_saved_checkpoint = latest_checkpoint
            
            metrics_history.append({
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'checkpoints': checkpoints
            })
            
            # Wait before next check
            time.sleep(30)
            
        except KeyboardInterrupt:
            log_message("\n\n⏸️  Monitoring stopped by user")
            break
        except Exception as e:
            log_message(f"⚠️  Error in monitoring: {e}")
            time.sleep(30)
    
    # Final summary
    log_message("\n" + "="*80)
    log_message("✅ TRAINING MONITORING COMPLETE")
    log_message("="*80)
    
    if metrics_history:
        log_message("\n📊 TRAINING SUMMARY:")
        log_message(f"  • Monitoring Duration: {len(metrics_history) * 30} seconds")
        log_message(f"  • Samples Collected: {len(metrics_history)}")
        
        # Get final status
        stdout, _, _ = run_remote_cmd("tail -5 /root/odia_ocr/training_main.log")
        if stdout:
            log_message(f"\n📋 Final Log Output:")
            for line in stdout.split('\n')[-5:]:
                if line.strip():
                    log_message(f"   {line}")
    
    log_message("\n🎉 Check HuggingFace for uploaded checkpoints!")
    log_message("   https://huggingface.co/shantipriya/qwen2.5-odia-ocr-v2")

if __name__ == "__main__":
    try:
        monitor_training()
    except Exception as e:
        log_message(f"❌ Fatal error: {e}")
        sys.exit(1)
