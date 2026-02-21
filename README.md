# Odia OCR

This repository contains code and scripts for fine-tuning and evaluating a Qwen Vision-Language model for Odia OCR.

Contents
- `training_ocr_qwen.py` — dataset handling, processor/collator, Trainer setup (main guarded).
- `run_forward.py` — quick single-forward sanity check with PEFT applied.
- `run_trainer_smoke.py` — smoke training run (few steps) for debugging.
- `run_eval.py` — wrapper that calls `Trainer.evaluate()` and prints metrics.
- `eval_qwen_ocr.py` — inference loop that computes CER/WER using `jiwer`.

Quickstart

1. Create and activate a Python venv, install dependencies:

```bash
python3 -m venv /root/venv
source /root/venv/bin/activate
pip install -r requirements.txt
```

If you don't have `requirements.txt`, install the main packages:

```bash
pip install torch transformers datasets peft jiwer pillow tqdm
```

2. Run a forward sanity check (on the remote GPU):

```bash
# on remote host, from project root
cd /root/odia_ocr
source /root/venv/bin/activate
python3 run_forward.py | tee run_forward.log
```

3. Run a tiny smoke training to verify Trainer loop:

```bash
python3 run_trainer_smoke.py 2>&1 | tee run_smoke.log
```

4. Evaluate with `Trainer.evaluate()`:

```bash
python3 run_eval.py 2>&1 | tee run_eval.log
```

5. Compute OCR accuracy (CER/WER):

```bash
python3 eval_qwen_ocr.py 2>&1 | tee eval_qwen.log
```

Notes and tips
- The fine-tuned model weights / adapter should be placed in `./qwen_ocr_finetuned` on the remote host. The evaluation script will try to load processor from that folder and fall back to the base model if needed. If your adapter-only checkpoint is used, make sure it contains `adapter_config.json` so PEFT can apply it.
- For debugging device placement, scripts currently force single-device loads (cuda:0) during quick checks; for full multi-GPU training use `accelerate` and proper input dispatching.
- If you encounter NaNs during training, try disabling fp16 or adjusting gradient clipping / LR / loss scaling.

Contact
- If you want, I can push these updates to https://github.com/shantipriyap/Odia-OCR — authorize/push credentials are required on your side.