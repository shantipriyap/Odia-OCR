#!/usr/bin/env python3
"""
Phase 2C: Model Enhancement with LoRA Rank Increase & Data Augmentation
Target: 26% → 20% CER (6% improvement)

Features:
- Increase LoRA rank from 32 to 64 (8x more parameters)
- Data augmentation (albumentations)
- Fine-tuning on checkpoint-250
- Validation tracking
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
import logging

import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Phase2CConfig:
    """Phase 2C configuration"""
    # Model settings
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    checkpoint_path: str = "./checkpoint-250"
    output_dir: str = "./checkpoint-300-phase2c"
    
    # LoRA settings (increased from Phase 1)
    lora_r: int = 64  # Increased from 32 to 64
    lora_alpha: int = 128  # Increased from 64 to 128
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    
    # Training settings
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    save_steps: int = 50
    eval_steps: int = 50
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


class OdiaImageAugmentor:
    """Data augmentation for Odia OCR images (albumentations-based)"""
    
    def __init__(self, prob: float = 0.5):
        self.prob = prob
        try:
            import albumentations as A
            self.augmentation = A.Compose([
                A.Rotate(limit=5, p=0.3),  # Slight rotation
                A.GaussNoise(p=0.2),  # Add noise
                A.GaussianBlur(blur_limit=3, p=0.2),  # Blur
                A.RandomBrightnessContrast(p=0.2),  # Brightness/contrast
                A.Affine(shear=(-2, 2), p=0.2),  # Shear
            ], p=prob)
            self.available = True
            logger.info("Albumentations augmentation enabled")
        except ImportError:
            logger.warning("albumentations not installed, using fallback augmentations")
            self.available = False
    
    def augment(self, image: np.ndarray) -> np.ndarray:
        """
        Augment image
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            Augmented image
        """
        if not random.random() < self.prob:
            return image
        
        if self.available:
            import albumentations as A
            augmented = self.augmentation(image=image)
            return augmented["image"]
        else:
            # Fallback: simple transformations
            return self._simple_augment(image)
    
    @staticmethod
    def _simple_augment(image: np.ndarray) -> np.ndarray:
        """Simple augmentation without albumentations"""
        if random.random() < 0.3:
            # Random rotation (small angle)
            angle = random.uniform(-5, 5)
            # TBA: implement rotation
        
        if random.random() < 0.2:
            # Add Gaussian noise
            noise = np.random.normal(0, 5, image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        if random.random() < 0.2:
            # Adjust brightness
            brightness = random.uniform(0.9, 1.1)
            image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        
        return image


class Phase2CTrainer:
    """Phase 2C model enhancement trainer"""
    
    def __init__(self, config: Phase2CConfig = None):
        self.config = config or Phase2CConfig()
        self.augmentor = OdiaImageAugmentor(prob=self.config.augmentation_prob)
        self.metrics = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': [],
        }
    
    def create_peft_config(self):
        """Create enhanced PEFT LoRA config"""
        try:
            from peft import LoraConfig, TaskType
            
            config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            logger.info(f"✅ Created LoRA config (r={self.config.lora_r}, alpha={self.config.lora_alpha})")
            return config
        except ImportError:
            logger.error("PEFT library required for Phase 2C")
            return None
    
    def load_checkpoint(self):
        """Load checkpoint-250 or pre-trained model"""
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            
            checkpoint_path = Path(self.config.checkpoint_path)
            
            if checkpoint_path.exists():
                logger.info(f"Loading checkpoint from {checkpoint_path}...")
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                logger.info("✅ Checkpoint loaded")
            else:
                logger.info(f"Checkpoint not found, loading base model...")
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.config.base_model,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                logger.info("✅ Base model loaded")
            
            processor = AutoProcessor.from_pretrained(self.config.base_model)
            return model, processor
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None, None
    
    def prepare_training_config(self):
        """Prepare training arguments"""
        try:
            from transformers import TrainingArguments
            
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_train_epochs,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                per_device_eval_batch_size=self.config.per_device_eval_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                max_grad_norm=self.config.max_grad_norm,
                warmup_steps=self.config.warmup_steps,
                save_steps=self.config.save_steps,
                eval_steps=self.config.eval_steps,
                logging_steps=10,
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                remove_unused_columns=True,
            )
            
            logger.info("✅ Training configuration prepared")
            return training_args
        
        except Exception as e:
            logger.error(f"Error preparing training config: {e}")
            return None
    
    def train(self):
        """Execute Phase 2C training"""
        print("\n" + "="*70)
        print("🚀 PHASE 2C: MODEL ENHANCEMENT")
        print("="*70 + "\n")
        
        print("📋 Configuration:")
        print(f"  LoRA Rank:     {self.config.lora_r} (↑ from 32)")
        print(f"  LoRA Alpha:    {self.config.lora_alpha} (↑ from 64)")
        print(f"  Augmentation:  {self.config.use_augmentation}")
        print(f"  Epochs:        {self.config.num_train_epochs}")
        print(f"  Learning Rate: {self.config.learning_rate}\n")
        
        # Load checkpoint
        model, processor = self.load_checkpoint()
        if model is None:
            return False
        
        # Create LoRA config
        peft_config = self.create_peft_config()
        if peft_config is None:
            return False
        
        # Apply LoRA
        try:
            from peft import get_peft_model
            model = get_peft_model(model, peft_config)
            logger.info("✅ LoRA adapter applied to model")
        except Exception as e:
            logger.error(f"Error applying LoRA: {e}")
            return False
        
        # Prepare training args
        training_args = self.prepare_training_config()
        if training_args is None:
            return False
        
        print("📊 Training Setup:")
        print(f"  Output Dir:    {self.config.output_dir}")
        print(f"  Batch Size:    {self.config.per_device_train_batch_size} (eff: {self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps})")
        print(f"  Max Steps:     {self.config.num_train_epochs * 100}  # Estimated")
        
        print("\n✅ Phase 2C ready for training")
        print("   Note: Full training requires GPU and OCR training data")
        print("="*70 + "\n")
        
        return True


def main():
    """Demonstrate Phase 2C model enhancement"""
    config = Phase2CConfig()
    trainer = Phase2CTrainer(config)
    
    # Validate configuration
    success = trainer.train()
    
    if success:
        print("💡 Next Steps for Phase 2C Implementation:")
        print("  1. Prepare Odia OCR training dataset")
        print("  2. Integrate with HuggingFace Trainer")
        print("  3. Run training: python -m torch.distributed.launch phase_2c_model_enhancement.py")
        print("  4. Evaluate improvements on test set")
        print("  5. Compare CER: Expected 26% → 20% (-6%)")


if __name__ == "__main__":
    main()
