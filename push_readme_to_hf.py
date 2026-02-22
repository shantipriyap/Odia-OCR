#!/usr/bin/env python3
"""
Upload local README.md as the model card on Hugging Face.

Usage:
    export HF_TOKEN=your_token
    python3 push_readme_to_hf.py --repo shantipriya/qwen2.5-odia-ocr-v2
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, HfFolder

DEFAULT_REPO_ID = "shantipriya/qwen2.5-odia-ocr"
README_PATH = Path("README.md")


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload README.md to a HF model repo")
    parser.add_argument(
        "--repo",
        default=os.getenv("HF_REPO", DEFAULT_REPO_ID),
        help="Target HF repo id (default: env HF_REPO or shantipriya/qwen2.5-odia-ocr)",
    )
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN") or HfFolder.get_token()
    if not token:
        print("ERROR: HF_TOKEN not found. Set HF_TOKEN or login via huggingface-cli.")
        return 1

    if not README_PATH.exists():
        print(f"ERROR: README not found at {README_PATH}")
        return 1

    api = HfApi(token=token)

    content = README_PATH.read_text(encoding="utf-8")
    api.upload_file(
        path_or_fileobj=content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
        commit_message="Update model card from local README"
    )

    print(f"README.md uploaded to https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
