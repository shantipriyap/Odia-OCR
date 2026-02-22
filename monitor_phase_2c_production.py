#!/usr/bin/env python3
"""
Phase 2C Training Monitor
Real-time monitoring of Phase 2C production training on GPU
"""

import subprocess
import json
import time
from datetime import datetime
from pathlib import Path


def get_gpu_status():
    """Get current GPU status"""
    try:
        result = subprocess.run(
            ['ssh', 'root@95.216.229.232', 'nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 4:
                return {
                    'gpu_util': f"{parts[0]}%",
                    'mem_util': f"{parts[1]}%",
                    'mem_used': f"{float(parts[2]):.1f}GB",
                    'mem_total': f"{float(parts[3]):.1f}GB"
                }
    except Exception as e:
        print(f"Error getting GPU status: {e}")
    
    return None


def get_training_log_tail(lines=30):
    """Get last N lines of training log"""
    try:
        result = subprocess.run(
            ['ssh', 'root@95.216.229.232', f'tail -{lines} /root/odia_ocr/phase_2c_training.log 2>/dev/null'],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout
    except Exception as e:
        print(f"Error getting log: {e}")
        return ""


def check_process_running():
    """Check if training process is running"""
    try:
        result = subprocess.run(
            ['ssh', 'root@95.216.229.232', 'ps aux | grep python3 | grep phase_2c | grep -v grep'],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except:
        return False


def monitor_phase_2c(interval=60, iterations=None):
    """
    Monitor Phase 2C training
    
    Args:
        interval: Seconds between checks
        iterations: Number of checks (None = infinite)
    """
    print("=" * 80)
    print("🚀 PHASE 2C PRODUCTION TRAINING MONITOR")
    print("=" * 80)
    
    check_count = 0
    
    while True:
        check_count += 1
        
        if iterations and check_count > iterations:
            print("\n✅ Monitoring completed")
            break
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] Check #{check_count}")
        print("-" * 80)
        
        # Check if process is running
        running = check_process_running()
        print(f"Process Status: {'🟢 RUNNING' if running else '🔴 STOPPED'}")
        
        # GPU Status
        gpu_status = get_gpu_status()
        if gpu_status:
            print(f"\n📊 GPU Status:")
            print(f"  GPU Utilization: {gpu_status['gpu_util']}")
            print(f"  Memory Util:     {gpu_status['mem_util']}")
            print(f"  Memory Used:     {gpu_status['mem_used']} / {gpu_status['mem_total']}")
        
        # Training Log
        print(f"\n📝 Training Log (last 15 lines):")
        log_tail = get_training_log_tail(15)
        if log_tail:
            for line in log_tail.strip().split('\n'):
                if line.strip():
                    print(f"  {line}")
        else:
            print("  (No log output yet)")
        
        if iterations and check_count >= iterations:
            break
        
        print(f"\n⏱️  Next check in {interval} seconds...")
        time.sleep(interval)


def print_summary():
    """Print Phase 2C training summary"""
    print("\n" + "=" * 80)
    print("📋 PHASE 2C TRAINING SUMMARY")
    print("=" * 80)
    print("""
🎯 Target:
   - Improvement: 32% → 26% CER (6% gain via Phase 2B) + 26% → 20% CER (6% gain via Phase 2C)
   - Total: 20% CER through Phase 2C
   - Timeline: ~7 days on A100

📊 Configuration:
   - Model: Qwen2.5-VL-3B-Instruct
   - Base: checkpoint-250
   - Output: checkpoint-300-phase2c
   - LoRA: r=64, alpha=128 (enhanced from Phase 1)
   - Batch Size: 1 (effective: 4 with gradient accumulation)
   - Learning Rate: 1e-4
   - Epochs: 3

💾 Data:
   - Source: HuggingFace (Odia OCR datasets)
   - Initial validation: 1000 samples
   - Full training: ~2000+ samples

🔧 Monitoring:
   - Training log: /root/odia_ocr/phase_2c_training.log
   - GPU metrics: Real-time nvidia-smi
   - Checkpoints: Saved every 50 steps
   - Max keep: 3 best checkpoints

📈 Expected Outcomes:
   - Checkpoint-300: Target 20% CER
   - Relative improvement: ~6% from Phase 2B baseline
   - Model ready for Phase 3 continuation

🚀 Next Phase (After 2C):
   - Phase 3: Full training resumption
   - Goal: 20% → 15% CER (5% additional improvement)
   - Expected duration: 3-4 days
""")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        print_summary()
    else:
        # Default: monitor every 60 seconds
        monitor_iterations = int(sys.argv[1]) if len(sys.argv) > 1 else None
        monitor_phase_2c(interval=60, iterations=monitor_iterations)
