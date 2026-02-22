#!/usr/bin/env python3
"""
Odia OCR Phase 2C Training - Live Status Dashboard
Real-time monitoring and update reporting
"""

import subprocess
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple


class Phase2CMonitor:
    """Monitor Phase 2C training with live status updates"""
    
    GPU_IP = "95.216.229.232"
    LOG_PATH = "/root/odia_ocr/phase_2c_training.log"
    CHECKPOINT_DIR = "/root/odia_ocr/checkpoint-300-phase2c"
    
    def __init__(self):
        self.last_check_time = None
        self.last_log_lines = 0
        self.metrics = {
            'start_time': None,
            'last_update': None,
            'training_active': False,
            'total_steps': 0,
            'current_loss': None,
            'checkpoints_saved': 0,
            'gpu_util': 0,
            'memory_used': 0,
            'errors': [],
        }
    
    def ssh_command(self, cmd: str) -> Tuple[str, int]:
        """Execute command on GPU server"""
        try:
            result = subprocess.run(
                ['ssh', f'root@{self.GPU_IP}', cmd],
                capture_output=True,
                text=True,
                timeout=15
            )
            return result.stdout, result.returncode
        except subprocess.TimeoutExpired:
            return "", 1
        except Exception as e:
            return f"Error: {e}", 1
    
    def check_training_status(self) -> Dict:
        """Check if training is active"""
        output, code = self.ssh_command("ps aux | grep phase_2c_production | grep python | grep -v grep | wc -l")
        is_running = code == 0 and int(output.strip()) > 0
        
        return {
            'active': is_running,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_gpu_metrics(self) -> Dict:
        """Get current GPU metrics"""
        cmd = "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits"
        output, code = self.ssh_command(cmd)
        
        if code == 0 and output.strip():
            try:
                parts = output.strip().split(', ')
                return {
                    'gpu_util': f"{parts[0]}%",
                    'mem_util': f"{parts[1]}%",
                    'mem_used': f"{float(parts[2]):.1f}GB",
                    'mem_total': f"{float(parts[3]):.1f}GB",
                    'temperature': f"{parts[4]}°C",
                    'available': True
                }
            except:
                return {'available': False}
        return {'available': False}
    
    def get_training_log_summary(self) -> Dict:
        """Parse training log for key metrics"""
        cmd = f"tail -100 {self.LOG_PATH} 2>/dev/null"
        output, code = self.ssh_command(cmd)
        
        summary = {
            'total_lines': len(output.split('\n')) if output else 0,
            'last_update': datetime.now().isoformat(),
            'has_errors': False,
            'key_messages': [],
            'loss_values': [],
        }
        
        if output:
            lines = output.split('\n')
            for line in lines[-20:]:  # Check last 20 lines
                if 'ERROR' in line or 'error' in line:
                    summary['has_errors'] = True
                    summary['key_messages'].append(f"⚠️ {line.strip()}")
                elif 'Step' in line or 'loss' in line or 'Loss' in line:
                    summary['key_messages'].append(line.strip())
                elif 'Checkpoint' in line or 'checkpoint' in line:
                    summary['key_messages'].append(f"💾 {line.strip()}")
        
        return summary
    
    def check_checkpoint_progress(self) -> Dict:
        """Check checkpoint creation progress"""
        cmd = f"ls -lh {self.CHECKPOINT_DIR}/ 2>/dev/null | grep -E 'checkpoint|adapter' | wc -l"
        output, code = self.ssh_command(cmd)
        
        try:
            count = int(output.strip())
            return {
                'checkpoint_count': count,
                'created': count > 0,
                'last_check': datetime.now().isoformat()
            }
        except:
            return {'checkpoint_count': 0, 'created': False}
    
    def print_live_status(self):
        """Print formatted live status"""
        status = self.check_training_status()
        gpu = self.get_gpu_metrics()
        log_summary = self.get_training_log_summary()
        checkpoint = self.check_checkpoint_progress()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("\n" + "="*80)
        print(f"🚀 PHASE 2C TRAINING LIVE STATUS - {timestamp}")
        print("="*80)
        
        # Training Status
        status_icon = "🟢 ACTIVE" if status['active'] else "🔴 INACTIVE"
        print(f"\n📊 Training Status: {status_icon}")
        
        # GPU Metrics
        if gpu.get('available'):
            print(f"\n💻 GPU Metrics:")
            print(f"   GPU Util:    {gpu['gpu_util']}")
            print(f"   Memory:      {gpu['mem_used']} / {gpu['mem_total']} ({gpu['mem_util']})")
            print(f"   Temp:        {gpu['temperature']}")
        else:
            print(f"\n💻 GPU Metrics: ⚠️ Unable to retrieve")
        
        # Checkpoint Progress
        print(f"\n💾 Checkpoint Progress:")
        print(f"   Files Created: {checkpoint['checkpoint_count']}")
        
        # Recent Log Messages
        if log_summary['key_messages']:
            print(f"\n📝 Recent Activity (Last 20 lines):")
            for msg in log_summary['key_messages'][-10:]:
                print(f"   {msg[:75]}")
        
        # Error Status
        if log_summary['has_errors']:
            print(f"\n⚠️  ERRORS DETECTED - Check logs immediately!")
        
        print("\n" + "="*80)
    
    def get_status_json(self) -> Dict:
        """Get status as JSON for programmatic access"""
        status = self.check_training_status()
        gpu = self.get_gpu_metrics()
        log_summary = self.get_training_log_summary()
        checkpoint = self.check_checkpoint_progress()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'training': status,
            'gpu': gpu,
            'log': log_summary,
            'checkpoint': checkpoint,
        }
    
    def save_status_snapshot(self):
        """Save current status to JSON file"""
        status = self.get_status_json()
        snapshot_file = Path("phase_2c_status_snapshot.json")
        
        with open(snapshot_file, 'w') as f:
            json.dump(status, f, indent=2)
        
        return snapshot_file
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report"""
        status = self.check_training_status()
        gpu = self.get_gpu_metrics()
        checkpoint = self.check_checkpoint_progress()
        
        report = f"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                     PHASE 2C TRAINING STATUS REPORT                            ║
╚════════════════════════════════════════════════════════════════════════════════╝

📅 TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🎯 PHASE 2C OBJECTIVE:
   • Model: Qwen2.5-VL-3B-Instruct
   • Base Checkpoint: checkpoint-250
   • Output: checkpoint-300-phase2c
   • Target CER: 20% (from 26%)
   • LoRA: rank 64, alpha 128 (0.39% trainable)
   • Expected Duration: 7-10 days
   • Dataset: 145,000+ Odia OCR samples

📊 CURRENT STATUS:
   Training Active: {'✅ YES' if status['active'] else '❌ NO'}
   Last Check: {datetime.now().isoformat()}

💻 GPU RESOURCES:
   Server: 95.216.229.232 (A100-SXM4-80GB)
   Status: {'✅ Available' if gpu.get('available') else '❌ Unable to retrieve'}
   
   GPU Utilization: {gpu.get('gpu_util', 'N/A')}
   Memory Usage:    {gpu.get('mem_used', 'N/A')} / {gpu.get('mem_total', 'N/A')} ({gpu.get('mem_util', 'N/A')})
   Temperature:     {gpu.get('temperature', 'N/A')}

💾 CHECKPOINT PROGRESS:
   Files Created: {checkpoint['checkpoint_count']}
   Output Dir: {self.CHECKPOINT_DIR}

📝 TRAINING LOG:
   Location: {self.LOG_PATH}
   Command: ssh root@95.216.229.232 "tail -50 {self.LOG_PATH}"

🔄 NEXT PHASE:
   Trigger: checkpoint-300-phase2c creation
   Phase 3: Full training resumption (3-4 days)
   Target: 20% → 15% CER

📋 MONITORING COMMANDS:
   Real-time: python3 monitor_and_report.py
   Snapshot:  python3 monitor_and_report.py snapshot
   Report:    python3 monitor_and_report.py report

════════════════════════════════════════════════════════════════════════════════════
"""
        return report


def main():
    """Run monitoring with various reporting modes"""
    import sys
    
    monitor = Phase2CMonitor()
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "snapshot":
            file = monitor.save_status_snapshot()
            print(f"✅ Status snapshot saved to {file}")
            print(json.dumps(monitor.get_status_json(), indent=2))
        
        elif mode == "report":
            print(monitor.generate_summary_report())
        
        elif mode == "json":
            print(json.dumps(monitor.get_status_json(), indent=2))
        
        elif mode == "loop":
            # Continuous monitoring
            iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            interval = int(sys.argv[3]) if len(sys.argv) > 3 else 60
            
            for i in range(1, iterations + 1):
                monitor.print_live_status()
                if i < iterations:
                    print(f"\n⏱️  Next check in {interval} seconds (press Ctrl+C to stop)...")
                    time.sleep(interval)
    else:
        # Default: single status + report
        monitor.print_live_status()
        print("\n" + monitor.generate_summary_report())


if __name__ == "__main__":
    main()
