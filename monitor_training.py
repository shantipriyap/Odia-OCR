#!/usr/bin/env python3
"""
Monitor and manage remote training on Odia OCR merged dataset
"""

import subprocess
import time
import sys

def run_remote_command(cmd):
    """Run command on remote server"""
    full_cmd = f"ssh root@135.181.8.206 '{cmd}'"
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=30)
    return result.stdout, result.stderr, result.returncode

def start_training():
    """Start training on remote server"""
    print("\n" + "="*70)
    print("🚀 STARTING REMOTE TRAINING")
    print("="*70 + "\n")
    
    # Kill any existing training
    print("Stopping any existing training processes...")
    run_remote_command("pkill -9 -f training_improved_merged || true")
    time.sleep(2)
    
    # Start new training
    print("Starting new training...")
    cmd = "cd /root/odia_ocr && source /root/venv/bin/activate && python3 training_improved_merged_dataset.py > training.log 2>&1 &"
    stdout, stderr, code = run_remote_command(cmd)
    print("✅ Training process started\n")
    
    # Wait for training to initialize
    print("Waiting for training to initialize (20 seconds)...")
    time.sleep(20)
    
    # Check logs
    print("\nChecking training logs...")
    stdout, stderr, code = run_remote_command("tail -100 /root/odia_ocr/training.log")
    print(stdout)
    if stderr:
        print(f"Errors: {stderr}")

def monitor_training():
    """Monitor ongoing training"""
    print("\n" + "="*70)
    print("📊 MONITORING TRAINING")
    print("="*70 + "\n")
    
    while True:
        # Check if process is running
        stdout, stderr, code = run_remote_command("pgrep -f training_improved_merged || echo 'STOPPED'")
        is_running = "STOPPED" not in stdout
        
        status_icon = "🟢 RUNNING" if is_running else "🔴 STOPPED"
        print(f"\n[{time.strftime('%H:%M:%S')}] {status_icon}")
        
        # Show last logs
        stdout, stderr, code = run_remote_command("tail -20 /root/odia_ocr/training.log")
        print(stdout[-500:] if len(stdout) > 500 else stdout)
        
        if not is_running:
            print("\n✅ Training completed!")
            break
        
        # Wait before next check
        print("\n(Checking again in 2 minutes...)")
        time.sleep(120)

if __name__ == "__main__":
    try:
        start_training()
        monitor_training()
    except KeyboardInterrupt:
        print("\n\n⏸️  Monitoring stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
