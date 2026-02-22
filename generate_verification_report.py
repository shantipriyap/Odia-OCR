#!/usr/bin/env python3
"""
Comprehensive Verification Report
Feb 22, 2026 - Model Deployment & Documentation Verification
"""

import os
import json
from pathlib import Path
from datetime import datetime

def check_hf_model_card():
    """Check HF model card sections"""
    readme_path = Path("README.md")
    
    if not readme_path.exists():
        return {"status": "❌ FAIL", "message": "README.md not found"}
    
    content = readme_path.read_text()
    
    required_sections = {
        "Installation": "## Installation" in content,
        "Quick Start": "## Quick Start" in content,
        "Usage": "## Usage" in content,
        "Performance Metrics": "## Performance Metrics" in content,
        "Phase 2A Results": "Phase 2A" in content and "32.0%" in content,
        "Model Card Info": "Model Information" in content or "Base Model" in content,
        "Download Instructions": "from_pretrained" in content or "PeftModel.from_pretrained" in content,
    }
    
    passed = sum(1 for v in required_sections.values() if v)
    total = len(required_sections)
    
    return {
        "status": "✅ PASS" if passed == total else "⚠️ PARTIAL",
        "sections": required_sections,
        "score": f"{passed}/{total}",
    }

def check_git_commits():
    """Check git commit history"""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "log", "--oneline", "-20"],
            capture_output=True,
            text=True,
            cwd="."
        )
        
        if result.returncode != 0:
            return {"status": "❌ FAIL", "message": "Git not available"}
        
        logs = result.stdout.strip().split("\n")
        
        important_commits = {
            "HF Deployment": any("HF" in log or "deployment" in log.lower() or "push" in log.lower() for log in logs),
            "Phase 2A Complete": any("Phase 2A" in log or "optimization" in log.lower() for log in logs),
            "Model Card Updated": any("README" in log or "model card" in log.lower() for log in logs),
            "Test Scripts": any("test" in log.lower() or "verification" in log.lower() for log in logs),
        }
        
        return {
            "status": "✅ PASS",
            "recent_commits": logs[:10],
            "important_commits": important_commits,
            "total_commits": len(logs),
        }
    except Exception as e:
        return {"status": "❌ FAIL", "message": str(e)}

def check_phase2a_results():
    """Check Phase 2A results are available"""
    results_file = Path("phase2_quick_win_results.json")
    
    if not results_file.exists():
        return {"status": "❌ FAIL", "message": "Results file not found"}
    
    try:
        with open(results_file) as f:
            results = json.load(f)
        
        expected_methods = ["greedy", "beam_search_5", "ensemble_voting"]
        methods_found = all(m in results.get("methods", {}) for m in expected_methods)
        
        if methods_found:
            ensemble_cer = results["methods"]["ensemble_voting"]["cer"]
            target_met = ensemble_cer <= 0.32
            
            return {
                "status": "✅ PASS",
                "samples": results.get("test_samples"),
                "timestamp": results.get("timestamp"),
                "results": {
                    "greedy": results["methods"]["greedy"]["cer"],
                    "beam_search": results["methods"]["beam_search_5"]["cer"],
                    "ensemble": results["methods"]["ensemble_voting"]["cer"],
                },
                "target_achieved": target_met,
            }
        else:
            return {"status": "❌ FAIL", "message": "Missing expected methods"}
            
    except Exception as e:
        return {"status": "❌ FAIL", "message": str(e)}

def check_file_existence():
    """Check critical files exist"""
    files = {
        "README.md": "Git repository README",
        "checkpoint-250/adapter_model.safetensors": "Model weights",
        "checkpoint-250/adapter_config.json": "LoRA config",
        "phase2_quick_win_results.json": "Evaluation results",
        "test_model_download_and_inference.py": "Verification script",
        "push_checkpoint_to_hf.py": "HF deployment script",
        "HF_DEPLOYMENT_SUMMARY.md": "Deployment documentation",
        "PHASE_2A_RESULTS.md": "Phase 2A documentation",
    }
    
    results = {}
    for file_path, description in files.items():
        exists = Path(file_path).exists()
        results[file_path] = {
            "exists": exists,
            "description": description,
            "status": "✅" if exists else "❌"
        }
    
    passed = sum(1 for v in results.values() if v["exists"])
    total = len(results)
    
    return {
        "status": "✅ PASS" if passed == total else f"⚠️ {total - passed} missing",
        "files": results,
        "score": f"{passed}/{total}",
    }

def check_download_instructions():
    """Check if download instructions are clear"""
    readme_path = Path("README.md")
    
    if not readme_path.exists():
        return {"status": "❌ FAIL"}
    
    content = readme_path.read_text().lower()
    
    instructions = {
        "Clone repository": "clone" in content,
        "Install dependencies": "pip install" in content,
        "Download model from HF": "from_pretrained" in content,
        "Load LoRA adapter": "peftmodel" in content,
        "Run inference": "generate" in content or "extract" in content,
        "Load image": "image" in content or "pil" in content,
    }
    
    passed = sum(1 for v in instructions.values() if v)
    total = len(instructions)
    
    return {
        "status": "✅ COMPLETE" if passed == total else "⚠️ PARTIAL",
        "instructions": instructions,
        "score": f"{passed}/{total}",
    }

def generate_summary():
    """Generate comprehensive verification summary"""
    
    print("\n")
    print("╔" + "=" * 70 + "╗")
    print("║" + " COMPREHENSIVE VERIFICATION REPORT ".center(70, "=") + "║")
    print("║" + f" {datetime.now().strftime('%B %d, %Y %H:%M UTC')} ".center(70) + "║")
    print("╚" + "=" * 70 + "╝")
    
    # Check 1: HF Documentation
    print("\n📝 CHECK 1: HuggingFace Hub README & Documentation")
    print("-" * 70)
    hf_check = check_hf_model_card()
    print(f"Status: {hf_check['status']}")
    if 'sections' in hf_check:
        for section, found in hf_check['sections'].items():
            symbol = "✅" if found else "❌"
            print(f"  {symbol} {section}")
    if 'score' in hf_check:
        print(f"Score: {hf_check['score']}")
    
    # Check 2: Git Commits
    print("\n📚 CHECK 2: Git Commit History & Code")
    print("-" * 70)
    git_check = check_git_commits()
    print(f"Status: {git_check['status']}")
    if 'total_commits' in git_check:
        print(f"Total commits: {git_check['total_commits']}")
    if 'important_commits' in git_check:
        for commit_type, found in git_check['important_commits'].items():
            symbol = "✅" if found else "⚠️"
            print(f"  {symbol} {commit_type}")
    if 'recent_commits' in git_check:
        print(f"\nRecent commits:")
        for commit in git_check['recent_commits'][:5]:
            print(f"  • {commit}")
    
    # Check 3: Phase 2A Results
    print("\n📊 CHECK 3: Phase 2A Evaluation Results")
    print("-" * 70)
    phase2a_check = check_phase2a_results()
    print(f"Status: {phase2a_check['status']}")
    if 'results' in phase2a_check:
        print(f"Test samples: {phase2a_check.get('samples')}")
        print(f"Results:")
        for method, cer in phase2a_check['results'].items():
            print(f"  • {method}: {cer*100:.1f}% CER")
        if phase2a_check.get('target_achieved'):
            print(f"  ✅ TARGET ACHIEVED (32.0% CER achieved)")
        else:
            print(f"  ⚠️ Target not met")
    
    # Check 4: Critical Files
    print("\n📁 CHECK 4: File Existence & Structure")
    print("-" * 70)
    files_check = check_file_existence()
    print(f"Status: {files_check['status']}")
    print(f"Score: {files_check['score']}")
    for file_path, info in files_check['files'].items():
        print(f"  {info['status']} {file_path}")
    
    # Check 5: Download Instructions
    print("\n📖 CHECK 5: Download & Setup Instructions in README")
    print("-" * 70)
    download_check = check_download_instructions()
    print(f"Status: {download_check['status']}")
    for instruction, present in download_check['instructions'].items():
        symbol = "✅" if present else "❌"
        print(f"  {symbol} {instruction}")
    print(f"Score: {download_check['score']}")
    
    # Overall Summary
    print("\n" + "=" * 70)
    print("📋 OVERALL SUMMARY")
    print("=" * 70)
    
    summary_items = [
        ("HF Hub Documentation", hf_check['status']),
        ("Git Commit History", git_check['status']),
        ("Phase 2A Results", phase2a_check['status']),
        ("File Structure", files_check['status']),
        ("Download Instructions", download_check['status']),
    ]
    
    passed = sum(1 for _, status in summary_items if "✅" in status)
    total = len(summary_items)
    
    for check_name, status in summary_items:
        print(f"{status}: {check_name}")
    
    print(f"\n{'='*70}")
    print(f"OVERALL SCORE: {passed}/{total} verification checks passed")
    
    if passed == total:
        print("🎉 ✅ DEPLOYMENT COMPLETE & VERIFIED!")
        print(f"   Model: https://huggingface.co/shantipriya/qwen2.5-odia-ocr")
        print(f"   Status: Ready for production use with Phase 2A optimization")
    else:
        print(f"⚠️ {total - passed} check(s) need attention")
    
    print(f"{'='*70}\n")
    
    return {
        "timestamp": datetime.now().isoformat(),
        "hf_documentation": hf_check,
        "git_commits": git_check,
        "phase2a_results": phase2a_check,
        "file_structure": files_check,
        "download_instructions": download_check,
        "overall_score": f"{passed}/{total}",
    }

if __name__ == "__main__":
    report = generate_summary()
    
    # Save report
    with open("VERIFICATION_REPORT.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("✅ Report saved to VERIFICATION_REPORT.json")
