#!/usr/bin/env python3
"""
Phase 2C: Production Training - BULLETPROOF VERSION
Directly loads from HuggingFace, no local dependencies
"""

from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
import torch
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

def main():
    print("\n" + "="*70)
    print("🚀 PHASE 2C: PRODUCTION TRAINING (BULLETPROOF)")
    print("="*70)
    print(f"Target: 26% → 20% CER (-6% improvement)\n")
    
    try:
        # ===== LOAD DATASETS =====
        print("📥 Loading Odia OCR datasets from HuggingFace...")
        datasets = []
        
        try:
            logger.info("Loading: OdiaGenAIOCR/Odia-lipi-ocr-data")
            ds1 = load_dataset("OdiaGenAIOCR/Odia-lipi-ocr-data")
            train_split = get_first_split(ds1)
            logger.info(f"✅ Loaded {len(train_split)} samples")
            datasets.append(train_split)
        except Exception as e:
            logger.error(f"Failed to load OdiaGenAIOCR: {e}")
            raise
        
        try:
            logger.info("Loading: tell2jyoti/odia-handwritten-ocr")
            ds2 = load_dataset("tell2jyoti/odia-handwritten-ocr")
            train_split_2 = get_first_split(ds2)
            logger.info(f"✅ Loaded {len(train_split_2)} samples")
            datasets.append(train_split_2)
        except Exception as e:
            logger.warning(f"Could not load handwritten dataset: {e}")
        
        if not datasets:
            raise ValueError("No datasets loaded!")
        
        # Combine
        dataset = concatenate_datasets(datasets)
        print(f"\n✅ DATASET READY: {len(dataset)} total samples\n")
        
        # ===== SETUP MODEL =====
        print("📦 Setting up model...")
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        checkpoint_path = "./checkpoint-250"
        
        # Load model
        try:
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info("✅ Checkpoint loaded")
        except Exception as e:
            logger.warning(f"Checkpoint not found, loading base model: {e}")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info("✅ Base model loaded")
        
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        # ===== APPLY LoRA =====
        print("\n🔧 Applying Enhanced LoRA...")
        lora_config = LoraConfig(
            r=64,              # Increased from 32
            lora_alpha=128,     # Increased from 64
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, lora_config)
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ LoRA applied (r=64, alpha=128)")
        logger.info(f"📊 Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        
        # ===== TRAINING CONFIG =====
        print("\n⚙️  Training Configuration:")
        print(f"  Epochs:        3")
        print(f"  Batch Size:    1 (effective: 4)")
        print(f"  Learning Rate: 1e-4")
        print(f"  Output Dir:    ./checkpoint-300-phase2c\n")
        
        training_args = TrainingArguments(
            output_dir="./checkpoint-300-phase2c",
            num_train_epochs=3,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            weight_decay=0.0,
            max_grad_norm=1.0,
            warmup_steps=100,
            save_steps=50,
            logging_steps=10,
            save_total_limit=3,
            remove_unused_columns=False,
            fp16=True,
            report_to=[],
        )
        
        # ===== TRAIN =====
        print("🔄 Initializing Trainer...")
        
        # Use default data collator if available
        try:
            from transformers import DataCollatorWithPadding
            collate_fn = DataCollatorWithPadding(processor.tokenizer)
        except:
            from transformers import default_data_collator
            collate_fn = default_data_collator
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )
        
        print("🟢 Starting training...\n")
        trainer.train()
        
        print("\n" + "="*70)
        print("✅ PHASE 2C TRAINING COMPLETE!")
        print("="*70)
        print(f"✅ Model saved to: ./checkpoint-300-phase2c")
        print(f"✅ Expected CER: 20% (from 26%)")
        print(f"📊 Next: Phase 3 (20% → 15% CER)\n")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ TRAINING FAILED: {e}", exc_info=True)
        print(f"\n❌ Phase 2C Training Failed")
        print(f"Error: {e}")
        return False


def get_first_split(ds):
    """Get first split from dataset"""
    if isinstance(ds, dict):
        return ds[list(ds.keys())[0]]
    return ds


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
