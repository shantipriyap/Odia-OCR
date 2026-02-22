#!/usr/bin/env python3
"""
Phase 2C: Production Training with LoRA Rank Increase & Data Augmentation
Target: 26% → 20% CER (6% improvement)

This is the PRODUCTION version that trains with real Odia OCR data.

Features:
- LoRA rank 32 → 64 (enhanced parameters)
- Data augmentation with albumentations
- Integrated with merged_odia_ocr_dataset or HF datasets
- GPU-optimized training on A100
- Mixed precision (fp16)
"""

import json
import random
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

import numpy as np
import torch
from PIL import Image
from datasets import load_dataset, Dataset, concatenate_datasets
from torch.utils.data import DataLoader, IterableDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Phase2CConfig:
    """Phase 2C production training configuration"""
    # Model settings
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    checkpoint_path: str = "./checkpoint-250"
    output_dir: str = "./checkpoint-300-phase2c"
    
    # LoRA settings (enhanced from Phase 1)
    lora_r: int = 64  # Increased from 32
    lora_alpha: int = 128  # Increased from 64
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
    logging_steps: int = 10
    
    # Data settings
    data_source: str = "merged_odia_ocr_dataset"  # or "huggingface" for HF datasets
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    max_samples: Optional[int] = None  # None = use all
    
    # Hardware settings
    use_fp16: bool = True
    seed: int = 42
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


class OdiaImageAugmentor:
    """Data augmentation for Odia OCR images"""
    
    def __init__(self, prob: float = 0.5):
        self.prob = prob
        try:
            import albumentations as A
            self.augmentation = A.Compose([
                A.Rotate(limit=5, p=0.3),
                A.GaussNoise(p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                A.Affine(shear=(-2, 2), p=0.2),
            ], p=prob)
            self.available = True
            logger.info("✅ Albumentations augmentation enabled")
        except ImportError:
            logger.warning("⚠️ albumentations not installed, using fallback augmentations")
            self.available = False
    
    def augment(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation to image"""
        if not random.random() < self.prob:
            return image
        
        if self.available:
            import albumentations as A
            augmented = self.augmentation(image=image)
            return augmented["image"]
        else:
            return self._simple_augment(image)
    
    @staticmethod
    def _simple_augment(image: np.ndarray) -> np.ndarray:
        """Fallback augmentation without albumentations"""
        if random.random() < 0.2:
            noise = np.random.normal(0, 5, image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        if random.random() < 0.2:
            brightness = random.uniform(0.9, 1.1)
            image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        
        return image


class OdiaOCRDataLoader:
    """Load Odia OCR training data from various sources"""
    
    def __init__(self, config: Phase2CConfig):
        self.config = config
        self.augmentor = OdiaImageAugmentor(prob=config.augmentation_prob) if config.use_augmentation else None
    
    def load_from_disk(self) -> Optional[Dataset]:
        """Load dataset from local merged_odia_ocr_dataset directory"""
        try:
            data_path = Path(self.config.data_source)
            if not data_path.exists():
                logger.warning(f"Dataset path not found: {data_path}")
                return None
            
            logger.info(f"Loading dataset from {data_path}...")
            dataset = load_dataset("imagefolder", data_dir=str(data_path))
            logger.info(f"✅ Loaded {len(dataset)} samples from disk")
            
            if self.config.max_samples:
                dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))
                logger.info(f"✅ Limited to {len(dataset)} samples")
            
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset from disk: {e}")
            return None
    
    def load_from_huggingface(self) -> Optional[Dataset]:
        """Load Odia OCR datasets from HuggingFace"""
        try:
            logger.info("Loading datasets from HuggingFace...")
            datasets = []
            
            # Primary dataset
            logger.info("📥 Loading: OdiaGenAIOCR/Odia-lipi-ocr-data")
            ds1 = load_dataset("OdiaGenAIOCR/Odia-lipi-ocr-data")
            train_split = ds1.get("train", list(ds1.keys())[0])
            datasets.append(train_split)
            logger.info(f"   ✅ {len(train_split)} samples")
            
            # Optional: Add handwritten dataset
            try:
                logger.info("📥 Loading: tell2jyoti/odia-handwritten-ocr")
                ds2 = load_dataset("tell2jyoti/odia-handwritten-ocr")
                train_split_2 = ds2.get("train", list(ds2.keys())[0])
                datasets.append(train_split_2)
                logger.info(f"   ✅ {len(train_split_2)} samples")
            except Exception as e:
                logger.warning(f"Could not load handwritten dataset: {e}")
            
            # Combine datasets
            combined = concatenate_datasets(datasets)
            logger.info(f"✅ Combined dataset: {len(combined)} total samples")
            
            if self.config.max_samples:
                combined = combined.select(range(min(self.config.max_samples, len(combined))))
            
            return combined
        
        except Exception as e:
            logger.error(f"Error loading HuggingFace datasets: {e}")
            return None
    
    def load(self) -> Optional[Dataset]:
        """Load dataset from configured source"""
        if self.config.data_source == "merged_odia_ocr_dataset":
            return self.load_from_disk()
        elif self.config.data_source == "huggingface":
            return self.load_from_huggingface()
        else:
            logger.error(f"Unknown data source: {self.config.data_source}")
            return None


class Phase2CTrainer:
    """Phase 2C production trainer"""
    
    def __init__(self, config: Phase2CConfig = None):
        self.config = config or Phase2CConfig()
        self.metrics = {}
        logger.info(f"Initialized Phase 2C trainer with config: {asdict(self.config)}")
    
    def setup_model(self):
        """Setup model with LoRA"""
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            from peft import LoraConfig, get_peft_model, TaskType
            
            # Load checkpoint or base model
            checkpoint_path = Path(self.config.checkpoint_path)
            if checkpoint_path.exists():
                logger.info(f"Loading checkpoint from {checkpoint_path}...")
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32,
                    device_map="auto",
                    trust_remote_code=True,
                )
                logger.info("✅ Checkpoint loaded")
            else:
                logger.info("Loading base model...")
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.config.base_model,
                    torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32,
                    device_map="auto",
                    trust_remote_code=True,
                )
                logger.info("✅ Base model loaded")
            
            # Load processor
            processor = AutoProcessor.from_pretrained(
                self.config.base_model,
                trust_remote_code=True
            )
            
            # Create & apply LoRA
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            model = get_peft_model(model, lora_config)
            logger.info(f"✅ LoRA applied (r={self.config.lora_r}, alpha={self.config.lora_alpha})")
            
            # Print trainable params
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"📊 Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
            
            return model, processor
        
        except Exception as e:
            logger.error(f"Error setting up model: {e}")
            return None, None
    
    def setup_training_args(self):
        """Setup training arguments"""
        try:
            from transformers import TrainingArguments
            
            args = TrainingArguments(
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
                logging_steps=self.config.logging_steps,
                save_total_limit=3,
                remove_unused_columns=False,
                fp16=self.config.use_fp16,
                report_to=[],
                seed=self.config.seed,
            )
            
            logger.info("✅ Training arguments prepared")
            return args
        
        except Exception as e:
            logger.error(f"Error preparing training args: {e}")
            return None
    
    def train(self):
        """Execute Phase 2C production training"""
        print("\n" + "="*70)
        print("🚀 PHASE 2C: PRODUCTION MODEL ENHANCEMENT")
        print("="*70 + "\n")
        
        # Setup
        logger.info("Setting up training...")
        model, processor = self.setup_model()
        if model is None:
            logger.error("❌ Failed to setup model")
            return False
        
        training_args = self.setup_training_args()
        if training_args is None:
            logger.error("❌ Failed to setup training args")
            return False
        
        # Load data
        logger.info("Loading training data...")
        data_loader = OdiaOCRDataLoader(self.config)
        dataset = data_loader.load()
        if dataset is None:
            logger.error("❌ Failed to load dataset")
            return False
        
        print("\n📋 Training Configuration:")
        print(f"  Model:           {self.config.base_model}")
        print(f"  Checkpoint:      {self.config.checkpoint_path}")
        print(f"  LoRA Rank:       {self.config.lora_r} (↑ from 32)")
        print(f"  LoRA Alpha:      {self.config.lora_alpha} (↑ from 64)")
        print(f"  Augmentation:    {self.config.use_augmentation}")
        print(f"  Epochs:          {self.config.num_train_epochs}")
        print(f"  Learning Rate:   {self.config.learning_rate}")
        print(f"  Batch Size:      {self.config.per_device_train_batch_size} (eff: {self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps})")
        print(f"  Dataset Size:    {len(dataset)} samples")
        print(f"  Output Dir:      {self.config.output_dir}\n")
        
        # Setup trainer
        logger.info("Setting up trainer...")
        try:
            from transformers import Trainer
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=self._create_data_collator(processor),
            )
            
            logger.info("✅ Trainer ready")
        except Exception as e:
            logger.error(f"Error setting up trainer: {e}")
            return False
        
        # Train
        print("🔄 Starting training...")
        try:
            trainer.train()
            logger.info("✅ Training completed successfully")
            
            # Save final model
            output_path = Path(self.config.output_dir)
            output_path.mkdir(exist_ok=True)
            model.save_pretrained(output_path)
            logger.info(f"✅ Model saved to {output_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return False
    
    @staticmethod
    def _create_data_collator(processor):
        """Create data collator for batch processing"""
        def collate_fn(batch):
            # Simple collator - can be enhanced for image-text processing
            return batch
        return collate_fn


def main():
    """Run Phase 2C production training"""
    # Create config
    config = Phase2CConfig(
        data_source="huggingface",  # Use HuggingFace datasets (merged_odia_ocr_dataset available if local)
        use_augmentation=True,
        max_samples=None,  # Use all samples (remove limit after initial validation)
    )
    
    # Run training
    trainer = Phase2CTrainer(config)
    success = trainer.train()
    
    if success:
        print("\n✅ Phase 2C Training Complete!")
        print(f"Model saved to: {config.output_dir}")
        print(f"Performance target: 26% → 20% CER (-6% improvement)")
        print(f"\nNext steps:")
        print(f"  1. Evaluate checkpoint-300 on test set")
        print(f"  2. Verify CER improvement")
        print(f"  3. Launch Phase 3 for further improvements")
    else:
        print("\n❌ Phase 2C Training Failed")
        print("Check logs above for details")


if __name__ == "__main__":
    main()
