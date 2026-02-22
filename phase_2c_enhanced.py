#!/usr/bin/env python3
"""
Phase 2C: Enhanced LoRA Training (Based on proven training_ocr_qwen.py)
Changes from Phase 1:
  - LoRA rank: 32 → 64 for more learning capacity
  - LoRA alpha: 64 → 128 for scaled adaptation
  - All else same as proven working configuration
"""

from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
import torch

print("="*70)
print("🚀 PHASE 2C: Enhanced LoRA Training")
print("="*70)
print("Target: 26% → 20% CER (improved from Phase 2B)\n")

# ===== LOAD DATASETS =====
def load_odia_datasets(use_multiple=True):
    """Load Odia OCR datasets"""
    datasets = []
    
    print("📥 Loading datasets...")
    print("  • OdiaGenAIOCR/Odia-lipi-ocr-data")
    ds1 = load_dataset("OdiaGenAIOCR/Odia-lipi-ocr-data")
    datasets.append(ds1["train"])
    print(f"    ✅ {len(ds1['train'])} samples")
    
    if use_multiple:
        try:
            print("  • tell2jyoti/odia-handwritten-ocr")
            ds2 = load_dataset("tell2jyoti/odia-handwritten-ocr")
            first_split = list(ds2.keys())[0]
            datasets.append(ds2[first_split])
            print(f"    ✅ {len(ds2[first_split])} samples")
        except Exception as e:
            print(f"    ⚠️  Could not load: {e}")
    
    combined = concatenate_datasets(datasets)
    print(f"\n✅ Combined: {len(combined)} samples\n")
    return combined

dataset = load_odia_datasets(use_multiple=True)

# ===== LOAD MODEL & PROCESSOR =====
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
print("📦 Loading model and processor...")
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# Image token detection
def find_image_token(processor):
    """Find or get image token"""
    tokenizer = processor.tokenizer
    for candidate in ["<image>", "<Image>", "<img>", "<image_patch>", "<image_token>"]:
        try:
            tid = tokenizer.convert_tokens_to_ids(candidate)
            if tid and tid != getattr(tokenizer, "unk_token_id", None):
                return candidate
        except:
            pass
    return "<image>"

image_token = find_image_token(processor)
if image_token not in processor.tokenizer.get_vocab():
    processor.tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})

print(f"✅ Model & processor ready\n")

# ===== ENHANCED LoRA CONFIG (Phase 2C) =====
print("🔧 Applying Enhanced LoRA:")
print(f"  Rank:       64 (↑ from 32)")
print(f"  Alpha:      128 (↑ from 64)")
print(f"  Dropout:    0.05\n")

lora_config = LoraConfig(
    r=64,                              # ENHANCED: was 32
    lora_alpha=128,                    # ENHANCED: was 64
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)

# ===== PREPROCESSING =====
def preprocess_function(example):
    """Preprocess examples"""
    image = example.get("image")
    if image is None:
        return None  # Skip None images
    try:
        image = image.convert("RGB")
    except:
        return None  # Skip bad images
    return {"image": image, "text": example.get("text", "")}

# Filter and preprocess
train_dataset = dataset.map(preprocess_function, batched=False).filter(lambda x: x is not None)

# ===== DATA COLLATOR =====
class QwenOCRDataCollator:
    """Custom data collator for Qwen2.5-VL"""
    
    def __init__(self, processor, image_token="<image>"):
        self.processor = processor
        self.image_token = image_token
        self.num_image_tokens = 1
    
    def __call__(self, batch):
        texts = []
        images = []
        
        for example in batch:
            image = example.get("image")
            text = example.get("text", "")
            
            # Prepend image placeholder
            text_with_image = f"{self.image_token}\n{text}"
            texts.append(text_with_image)
            images.append(image)
        
        # Process batch
        processed = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        return processed

collator = QwenOCRDataCollator(processor, image_token)

# ===== TRAINING CONFIG =====
print("⚙️  Training Configuration:")
print(f"  Base:       checkpoint-250")
print(f"  Output:     checkpoint-300-phase2c")
print(f"  Epochs:     1 (aggressive)")
print(f"  Batch:      1 (eff: 4)")
print(f"  LR:         1e-4\n")

training_args = TrainingArguments(
    output_dir="./checkpoint-300-phase2c",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    warmup_steps=50,
    save_steps=100,
    logging_steps=10,
    save_total_limit=3,
    fp16=True,
    report_to=[],
)

# ===== LOAD BASE MODEL =====
print("📥 Loading base model...")
try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "./checkpoint-250",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("✅ Checkpoint-250 loaded\n")
except:
    print("Loading base model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("✅ Base model loaded\n")

# ===== APPLY LoRA =====
model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"📊 LoRA Applied:")
print(f"  Trainable:  {trainable:,} / {total:,} ({100*trainable/total:.2f}%)\n")

# ===== TRAIN =====
print("🔄 Starting Training...")
print("="*70)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collator,
)

try:
    trainer.train()
    print("\n" + "="*70)
    print("✅ Phase 2C Training Complete!")
    print("="*70)
    print(f"✅ Model saved: checkpoint-300-phase2c")
    print(f"✅ Target CER: 20% (from 26%)")
    print(f"📊 Next: Phase 3 (20% → 15% CER)")
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
