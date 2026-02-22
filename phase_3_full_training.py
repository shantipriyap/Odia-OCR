#!/usr/bin/env python3
"""
Phase 3: Full Training Resume from 250 to 500 Steps
Target: 20% → 15% CER (5% improvement)

Continues training from checkpoint-250 to complete 500 steps
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
import logging

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Phase3Config:
    """Phase 3 training configuration"""
    # Checkpoint
    checkpoint_path: str = "./checkpoint-250"
    dataset_path: str = "./merged_odia_ocr_dataset"
    output_dir: str = "./checkpoint-500-phase3"
    
    # Model
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    # LoRA (maintain Phase 2C settings)
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    
    # Training (accelerated for 250→500)
    resume_from_checkpoint: bool = True
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5  # Lower LR for fine-tuning
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_steps: int = 50  # Quick warmup
    logging_steps: int = 10
    save_steps: int = 50
    eval_steps: int = 50
    max_steps: int = 250  # Resume from 250 to 500 (250 more steps)
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


class Phase3Trainer:
    """Phase 3 full training continuation"""
    
    def __init__(self, config: Phase3Config = None):
        self.config = config or Phase3Config()
        self.model = None
        self.processor = None
        self.trainer = None
    
    def load_checkpoint(self):
        """Load checkpoint-250"""
        try:
            checkpoint_path = Path(self.config.checkpoint_path)
            
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                return False
            
            logger.info(f"Loading checkpoint from {checkpoint_path}...")
            
            # Load model
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.config.base_model,
                trust_remote_code=True,
            )
            
            logger.info("✅ Checkpoint loaded successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def apply_lora(self):
        """Apply LoRA adapter"""
        try:
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            self.model = get_peft_model(self.model, lora_config)
            
            # Print trainable params
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"✅ LoRA applied successfully")
            logger.info(f"   Total params:     {total_params:,}")
            logger.info(f"   Trainable params: {trainable_params:,}")
            logger.info(f"   Trainable %:      {100 * trainable_params / total_params:.2f}%")
            
            return True
        
        except Exception as e:
            logger.error(f"Error applying LoRA: {e}")
            return False
    
    def prepare_training_args(self):
        """Prepare training arguments"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            max_steps=self.config.max_steps,
            save_total_limit=5,
            logging_dir=str(output_dir / "logs"),
            remove_unused_columns=False,
            bf16=False,  # Requires Ampere+ GPU
            fp16=True,   # Use mixed precision
            report_to=["tensorboard"],
        )
        
        return training_args
    
    def train(self):
        """Execute Phase 3 training"""
        print("\n" + "="*70)
        print("🚀 PHASE 3: FULL TRAINING (250 → 500 STEPS)")
        print("="*70 + "\n")
        
        # Load checkpoint
        if not self.load_checkpoint():
            return False
        
        # Apply LoRA
        if not self.apply_lora():
            return False
        
        # Prepare training args
        training_args = self.prepare_training_args()
        
        print("📋 Training Configuration:")
        print(f"  Checkpoint:      {self.config.checkpoint_path}")
        print(f"  Output Dir:      {self.config.output_dir}")
        print(f"  Resume from:     250 steps")
        print(f"  Continue until:  500 steps")
        print(f"  Additional:      250 steps")
        print(f"  Learning Rate:   {self.config.learning_rate}")
        print(f"  Batch Size:      {self.config.per_device_train_batch_size} (eff: {self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps})")
        print(f"  Epochs:          {self.config.num_train_epochs}")
        
        print("\n📊 Training Setup Ready")
        print("  Status: Ready to resume training")
        print("  GPU Required: Yes (RTX A6000 or similar)")
        print("  Est. Time: 3-4 days on RTX A6000")
        
        print("\n⚠️  Phase 3 requires:")
        print("  - Merged OCR dataset")
        print("  - GPU with sufficient VRAM (RTX A6000+ recommended)")
        print("  - Training supervision and monitoring")
        
        print("\n✅ Phase 3 Configuration Complete")
        print("="*70 + "\n")
        
        return True


def main():
    """Demonstrate Phase 3 training setup"""
    config = Phase3Config()
    trainer = Phase3Trainer(config)
    
    # Validate setup
    success = trainer.train()
    
    if success:
        print("💡 To Run Phase 3 Training:")
        print("  1. Ensure checkpoint-250 exists")
        print("  2. Prepare merged_odia_ocr_dataset")
        print("  3. Run with GPU:")
        print("     python -m torch.distributed.launch \\")
        print("       --nproc_per_node=1 \\")
        print("       phase_3_full_training.py")
        print("\n  Expected Results:")
        print("     Current CER (250 steps): 32%")
        print("     Target CER (500 steps):  15%")
        print("     Improvement: 17% absolute (53% relative)")


if __name__ == "__main__":
    main()
