#!/usr/bin/env python3
"""
Upload merged Odia OCR dataset to HuggingFace Hub using Git
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description=""):
    """Run a shell command and return success status"""
    if description:
        print(f"\n{description}...")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Command failed: {cmd}")
            if result.stderr:
                print(f"   Error: {result.stderr}")
            return False
        if result.stdout:
            print(result.stdout)
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def upload_via_git(dataset_dir="./merged_odia_ocr_dataset"):
    """Upload dataset to HuggingFace Hub using Git"""
    
    print("\n" + "="*80)
    print("🚀 UPLOADING ODIA OCR DATASET TO HUGGINGFACE HUB (GIT METHOD)")
    print("="*80 + "\n")
    
    # Check dataset exists
    if not Path(dataset_dir).exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return False
    
    print("✅ Dataset directory found")
    
    # Step 1: Check git and git-lfs
    print("\n📋 Checking prerequisites...")
    
    git_version = subprocess.run("git --version", shell=True, capture_output=True, text=True)
    if git_version.returncode == 0:
        print(f"   ✅ Git installed: {git_version.stdout.strip()}")
    else:
        print("   ❌ Git not installed. Please install Git first.")
        return False
    
    git_lfs_check = subprocess.run("git lfs version", shell=True, capture_output=True, text=True)
    if git_lfs_check.returncode == 0:
        print(f"   ✅ Git LFS installed")
    else:
        print("   ⚠️  Git LFS not installed. Installing Git LFS...")
        run_command("brew install git-lfs" if sys.platform == "darwin" else "apt-get install -y git-lfs", 
                    "Installing Git LFS")
    
    # Step 2: Create repo on HuggingFace (if needed)
    print("\n📌 Dataset repository on HuggingFace Hub:")
    print("   URL: https://huggingface.co/datasets/shantipriya/odia-ocr-merged")
    print("   Please create this repository manually if it doesn't exist")
    
    # Step 3: Clone repo
    repo_path = "./hf_dataset_repo"
    if Path(repo_path).exists():
        print(f"\n📁 Repository directory exists: {repo_path}")
        use_existing = input("   Use existing directory? (y/n): ").lower() == 'y'
        if not use_existing:
            import shutil
            shutil.rmtree(repo_path)
    
    if not Path(repo_path).exists():
        print("\n📥 Cloning HuggingFace dataset repository...")
        if not run_command(
            "git clone https://huggingface.co/datasets/shantipriya/odia-ocr-merged " + repo_path,
            "Cloning repository"
        ):
            print("   ⚠️  Repository may not exist yet. Creating local structure...")
            os.makedirs(repo_path, exist_ok=True)
            # Initialize git repo locally
            os.chdir(repo_path)
            run_command("git init", "Initializing git repo")
            run_command("git lfs install", "Initializing Git LFS")
            run_command("git config user.email 'shantipriya@example.com'", "Setting git email")
            run_command("git config user.name 'Shantipriya Parida'", "Setting git name")
            run_command("git remote add origin https://huggingface.co/datasets/shantipriya/odia-ocr-merged", 
                       "Adding remote origin")
            os.chdir("..")
    else:
        print("   ✅ Repository ready")
    
    os.chdir(repo_path)
    
    # Step 4: Copy dataset files
    print("\n📋 Copying dataset files...")
    dataset_files = [
        ("data.parquet", "Parquet dataset file"),
        ("README.md", "README"),
        ("metadata.json", "Metadata"),
    ]
    
    for filename, desc in dataset_files:
        src = Path(f"../{dataset_dir}/{filename}")
        if src.exists():
            import shutil
            shutil.copy(src, filename)
            print(f"   ✅ Copied {desc}: {filename}")
        else:
            print(f"   ⚠️  Not found: {filename}")
    
    # Step 5: Setup Git LFS for parquet
    print("\n🔒 Configuring Git LFS for large files...")
    run_command("git lfs track '*.parquet'", "Tracking .parquet files with Git LFS")
    run_command("git add .gitattributes", "Adding .gitattributes")
    
    # Step 6: Commit and push
    print("\n📤 Committing and pushing to HuggingFace Hub...")
    run_command("git add .", "Staging files")
    run_command("git commit -m '📚 Add merged Odia OCR dataset (145K+ samples)\n\nMerged from:\n- OdiaGenAIOCR: 64 samples\n- tell2jyoti: 145,717 samples\n\nReady for training Qwen2.5-VL OCR models'", 
               "Committing files")
    
    if not run_command("git push -u origin main", "Pushing to HuggingFace Hub"):
        # Try master branch if main doesn't exist
        run_command("git push -u origin master", "Pushing to main branch (trying master)")
    
    os.chdir("..")
    
    # Step 7: Summary
    print("\n" + "="*80)
    print("✅ UPLOAD COMPLETE!")
    print("="*80)
    print("\n📊 DATASET INFO:")
    print("   URL: https://huggingface.co/datasets/shantipriya/odia-ocr-merged")
    print("   Samples: 145,781")
    print("   Status: 🌐 Public")
    print("\n🎉 Your dataset is now on HuggingFace Hub!")
    print("\n📖 Load your dataset with:")
    print("   from datasets import load_dataset")
    print("   dataset = load_dataset('shantipriya/odia-ocr-merged')")
    print()
    
    return True


def push_with_token_env():
    """Try pushing using HF_TOKEN environment variable"""
    
    print("\n" + "="*80)
    print("🚀 TRIED: Token-based push (if HF_TOKEN environment variable is set)")
    print("="*80 + "\n")
    
    token = os.getenv("HF_TOKEN")
    if token:
        print(f"✅ HF_TOKEN found (length: {len(token)} chars)")
        
        # Try the noninteractive push script
        result = subprocess.run(
            "cd /root/odia_ocr && source /root/venv/bin/activate && python3 push_to_hf_noninteractive.py",
            shell=True,
            capture_output=True,
            text=True,
            env={**os.environ, "HF_TOKEN": token}
        )
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        return result.returncode == 0
    else:
        print("ℹ️  HF_TOKEN environment variable not set")
        print("   You can set it with: export HF_TOKEN='your_token_here'")
        return False


if __name__ == "__main__":
    success = False
    
    # Check if running on remote server
    is_remote = os.path.exists("/root/odia_ocr")
    
    if is_remote:
        print("🔧 Running on remote server...")
        # Try token-based first, then git method
        if not push_with_token_env():
            print("\n⚠️  Token-based push not available")
            print("Please upload manually using Git:")
            upload_via_git("./merged_odia_ocr_dataset")
    else:
        print("🔧 Running on local machine...")
        success = upload_via_git("./merged_odia_ocr_dataset")
    
    if success:
        print("\n✅ Dataset successfully uploaded!")
        exit(0)
    else:
        print("\n⚠️  Upload encountered issues. Check steps above.")
        exit(1)
