#!/usr/bin/env python3
"""Quick status check for Phase 2C training"""
import subprocess
import sys

def check_status():
    try:
        # Check if process running
        result = subprocess.run(
            'ssh root@95.216.229.232 "ps aux | grep python3 | grep -v grep"',
            shell=True,
            capture_output=True,
            text=True,
            timeout=5
        )
        
        print("🔍 Phase 2C Status Check\n")
        
        if "phase_2c" in result.stdout or "python3" in result.stdout:
            print("✅ Training process is RUNNING on GPU")
            print(f"   Processes: {len([l for l in result.stdout.split(chr(10)) if l.strip()])}")
        else:
            print("⚠️  No training process found")
            
        # Check log file size
        result2 = subprocess.run(
            'ssh root@95.216.229.232 "ls -lh /root/odia_ocr/phase_2c_training.log 2>/dev/null | awk \'{print $5}\'"',
            shell=True,
            capture_output=True,
            text=True,
            timeout=5
        )
        
        log_size = result2.stdout.strip()
        print(f"\n📄 Training Log: {log_size or 'Not found'}")
        
        # Quick GPU check
        result3 = subprocess.run(
            'ssh root@95.216.229.232 "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"',
            shell=True,
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result3.stdout.strip():
            parts = result3.stdout.strip().split(',')
            print(f"\n📊 GPU Status:")
            print(f"   Memory: {parts[0].strip()} MB / {parts[1].strip()} MB")
            print(f"   Utilization: {parts[2].strip()}%")
        
        print("\n" + "="*50)
        print("Monitoring script running in background...")
        print("To view full training updates:")
        print("   python3 monitor_phase_2c.py")
        print("="*50)
        
    except subprocess.TimeoutExpired:
        print("⏱️  SSH timeout - GPU server may be busy")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_status()
