#!/usr/bin/env python3
"""
Simple status monitor for Odia OCR training
Shows current progress and checkpoint status
"""

import subprocess
import re
from datetime import datetime

REMOTE_HOST = "135.181.8.206"
REMOTE_LOG = "/root/odia_ocr/training_main.log"
REMOTE_CKPT_DIR = "/root/odia_ocr/qwen_odia_ocr_improved_v2"

def ssh_cmd(cmd):
    """Run SSH command"""
    try:
        result = subprocess.run(
            ["ssh", f"root@{REMOTE_HOST}", cmd],
            capture_output=True,
            text=True,
            timeout=15
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except:
        return None

def get_progress():
    """Get training progress"""
    cmd = f"tail -20 {REMOTE_LOG} | grep -oE '[0-9]+%\\|.*\\| [0-9]+/[0-9]+' | tail -1"
    output = ssh_cmd(cmd)
    
    if output:
        # Parse: "60%|██████    | 300/500"
        match = re.search(r'(\d+)%.*?(\d+)/(\d+)', output)
        if match:
            percent = int(match.group(1))
            current = int(match.group(2))
            total = int(match.group(3))
            
            # Calculate time
            if current > 0:
                speed_match = re.search(r'([\d.]+)it/s', output)
                if speed_match:
                    speed = float(speed_match.group(1))
                    remaining_sec = (total - current) / speed
                    remaining_min = remaining_sec / 60
                    remaining_str = f"{int(remaining_min)}m" if remaining_min > 1 else f"{int(remaining_sec)}s"
                else:
                    remaining_str = "unknown"
            else:
                remaining_str = "unknown"
            
            return {
                "percent": percent,
                "current": current,
                "total": total,
                "remaining": remaining_str,
                "raw": output
            }
    return None

def get_checkpoints():
    """Get list of checkpoints"""
    cmd = f"ls -d {REMOTE_CKPT_DIR}/checkpoint-* 2>/dev/null | sort -V"
    output = ssh_cmd(cmd)
    
    if output:
        return [line.split('/')[-1] for line in output.split('\n') if line]
    return []

def main():
    print("\n" + "="*70)
    print("📊 ODIA OCR TRAINING STATUS")
    print("="*70)
    
    # Get progress
    progress = get_progress()
    if progress:
        bar_length = 50
        filled = int(bar_length * progress['percent'] / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        print(f"\n{progress['percent']:3d}% |{bar}| {progress['current']:3d}/{progress['total']} steps")
        print(f"⏱️  Remaining: ~{progress['remaining']}")
        print(f"📍 Raw: {progress['raw']}")
    else:
        print("⚠️  Could not fetch progress")
    
    # Get checkpoints
    checkpoints = get_checkpoints()
    if checkpoints:
        print(f"\n✅ Checkpoints Saved: {len(checkpoints)}")
        for ckpt in checkpoints:
            step = int(ckpt.replace('checkpoint-', ''))
            percent = (step / 500) * 100
            print(f"   • {ckpt} ({percent:.0f}%)")
    else:
        print("\n⚠️  No checkpoints found yet")
    
    print("\n" + "="*70)
    print("💡 Next: Run 'python3 upload_checkpoints.py' to push to HF Hub")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
