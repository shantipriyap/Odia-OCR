#!/usr/bin/env python3
"""
Real-time monitoring for Phase 2C training on GPU
Tracks: training loss, validation metrics, time, GPU memory, checkpoints
"""

import subprocess
import time
import re
from datetime import datetime
from pathlib import Path

GPU_SERVER = "root@95.216.229.232"
LOG_FILE = "/root/odia_ocr/phase_2c_training.log"
CHECK_INTERVAL = 60  # Check every 60 seconds


def run_ssh_command(cmd):
    """Execute SSH command and return output"""
    try:
        result = subprocess.run(
            f'ssh {GPU_SERVER} "{cmd}"',
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error: {e}"


def get_log_tail(lines=100):
    """Get last N lines of training log"""
    cmd = f"tail -{lines} {LOG_FILE}"
    return run_ssh_command(cmd)


def get_gpu_status():
    """Get current GPU usage"""
    cmd = "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader"
    output = run_ssh_command(cmd)
    try:
        parts = output.strip().split(',')
        if len(parts) >= 4:
            mem_used = parts[0].strip().split()[0]
            mem_total = parts[1].strip().split()[0]
            util = parts[2].strip()
            temp = parts[3].strip()
            return {
                'mem_used': mem_used,
                'mem_total': mem_total,
                'utilization': util,
                'temperature': temp
            }
    except:
        pass
    return None


def get_training_metrics(log_content):
    """Extract training metrics from log"""
    metrics = {
        'current_step': None,
        'total_steps': None,
        'loss': None,
        'learning_rate': None,
        'training_started': False,
        'errors': []
    }
    
    lines = log_content.split('\n')
    for line in lines[-50:]:  # Check last 50 lines
        # Check for training progress
        if 'Step' in line and 'loss' in line.lower():
            match = re.search(r'Step (\d+)/(\d+)', line)
            if match:
                metrics['current_step'] = int(match.group(1))
                metrics['total_steps'] = int(match.group(2))
            
            loss_match = re.search(r'loss[:\s]+([0-9.]+)', line, re.IGNORECASE)
            if loss_match:
                metrics['loss'] = float(loss_match.group(1))
        
        # Check for learning rate
        if 'lr' in line.lower():
            lr_match = re.search(r'lr[:\s]+([0-9.e+-]+)', line, re.IGNORECASE)
            if lr_match:
                metrics['learning_rate'] = lr_match.group(1)
        
        # Check for training started
        if 'training' in line.lower() and ('start' in line.lower() or 'begin' in line.lower()):
            metrics['training_started'] = True
        
        # Check for errors
        if 'error' in line.lower() or 'exception' in line.lower():
            metrics['errors'].append(line.strip())
    
    return metrics


def print_status(iteration, log_content, gpu_status, metrics):
    """Print formatted status update"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "="*80)
    print(f"🔍 PHASE 2C MONITORING UPDATE #{iteration}")
    print(f"⏰ {timestamp}")
    print("="*80)
    
    # GPU Status
    if gpu_status:
        print(f"\n📊 GPU Status:")
        print(f"   Memory: {gpu_status.get('mem_used', 'N/A')} / {gpu_status.get('mem_total', 'N/A')}")
        print(f"   Utilization: {gpu_status.get('utilization', 'N/A')}")
        print(f"   Temperature: {gpu_status.get('temperature', 'N/A')}")
    
    # Training Metrics
    print(f"\n📈 Training Metrics:")
    if metrics['current_step']:
        progress = (metrics['current_step'] / metrics['total_steps'] * 100) if metrics['total_steps'] else 0
        print(f"   Step: {metrics['current_step']}/{metrics['total_steps']} ({progress:.1f}%)")
    
    if metrics['loss'] is not None:
        print(f"   Loss: {metrics['loss']:.4f}")
    
    if metrics['learning_rate']:
        print(f"   Learning Rate: {metrics['learning_rate']}")
    
    print(f"   Status: {'🟢 Training' if metrics['training_started'] else '🟡 Initializing'}")
    
    # Errors
    if metrics['errors']:
        print(f"\n⚠️  Errors ({len(metrics['errors'])}):")
        for error in metrics['errors'][-3:]:  # Show last 3 errors
            print(f"   {error[:70]}...")
    
    # Recent log lines
    print(f"\n📝 Recent Log (last 5 lines):")
    recent_lines = [l for l in log_content.split('\n') if l.strip()][-5:]
    for line in recent_lines:
        short_line = line[:76]
        print(f"   {short_line}")
    
    print("="*80 + "\n")


def check_training_complete(log_content):
    """Check if training has completed"""
    if 'training complete' in log_content.lower() or 'finished' in log_content.lower():
        return True
    if '***** Running training *****' in log_content and 'evaluation' in log_content.lower():
        return True
    return False


def main():
    """Main monitoring loop"""
    iteration = 0
    last_step = -1
    
    print("🚀 Starting Phase 2C Monitoring...")
    print(f"📍 Server: {GPU_SERVER}")
    print(f"📄 Log: {LOG_FILE}")
    print(f"⏱️  Check interval: {CHECK_INTERVAL}s\n")
    
    try:
        while True:
            iteration += 1
            
            # Get current status
            log_content = get_log_tail(150)
            gpu_status = get_gpu_status()
            metrics = get_training_metrics(log_content)
            
            # Print update
            print_status(iteration, log_content, gpu_status, metrics)
            
            # Detect progress
            if metrics['current_step'] and metrics['current_step'] > last_step:
                last_step = metrics['current_step']
                time_per_step = CHECK_INTERVAL / ((metrics['current_step'] - last_step) or 1)
                if metrics['total_steps']:
                    eta_seconds = (metrics['total_steps'] - metrics['current_step']) * time_per_step
                    eta_hours = eta_seconds / 3600
                    print(f"⏳ ETA: ~{eta_hours:.1f} hours")
            
            # Check if complete
            if check_training_complete(log_content):
                print("\n✅ TRAINING COMPLETE!")
                print("Ready to proceed with Phase 3 full training.")
                break
            
            # Wait before next check
            print(f"⏸️  Next check in {CHECK_INTERVAL}s... (Ctrl+C to stop)\n")
            time.sleep(CHECK_INTERVAL)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped by user")
        print("Training continues on GPU. Resume monitoring anytime with:")
        print(f"   python3 monitor_phase_2c.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
