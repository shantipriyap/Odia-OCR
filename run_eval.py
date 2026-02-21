#!/usr/bin/env python3
"""Run evaluation with the existing collator / processor / dataset from
`training_ocr_qwen.py` and print Trainer evaluation metrics.

Usage (remote):
  /root/venv/bin/python3 run_eval.py
"""
from peft import get_peft_model
import torch
from transformers import Trainer

# Reuse objects from training_ocr_qwen (import-safe module)
import training_ocr_qwen as cfg

from transformers import logging
logging.set_verbosity_info()


def main():
    # Prefer a single CUDA device for deterministic placement during eval
    use_cuda = torch.cuda.is_available()

    device_map = {"": "cuda:0"} if use_cuda else None

    print("Loading model (this may take a while)...")
    model = cfg.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model_name,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map=device_map,
    )

    # If tokenizer was extended, resize embeddings
    model.resize_token_embeddings(len(cfg.processor.tokenizer))

    # Wrap with PEFT adapters (assumes cfg.lora_config exists)
    model = get_peft_model(model, cfg.lora_config)

    # Build Trainer reusing training args and collator
    trainer = Trainer(
        model=model,
        args=cfg.training_args,
        eval_dataset=cfg.eval_dataset,
        data_collator=cfg.data_collator,
    )

    print("Running evaluation...")
    metrics = trainer.evaluate()

    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
