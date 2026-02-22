#!/usr/bin/env python3
"""
Phase 2C: Training with Direct Data Handling
Simplified to avoid dataset format issues
"""

from datasets import load_dataset, concatenate_datasets
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    AutoProcessor,
)
from peft import LoraConfig, get_peft_model
import torch
import sys

print("\n" + "="*70)
print("🚀 PHASE 2C: Enhanced LoRA Training (Direct Mode)")
print("="*70 + "\n")

# ===== LOAD DATASETS =====
print("📥 Loading datasets...")
datasets = []

try:
    print("  • Primary: OdiaGenAIOCR/Odia-lipi-ocr-data")
    ds1 = load_dataset("OdiaGenAIOCR/Odia-lipi-ocr-data")
    datasets.append(ds1["train"])
    print(f"    ✅ {len(ds1['train'])} samples")
except Exception as e:
    print(f"    ❌ Failed: {e}")
    sys.exit(1)

try:
    print("  • Secondary: tell2jyoti/odia-handwritten-ocr")
    ds2 = load_dataset("tell2jyoti/odia-handwritten-ocr")
    first_split = list(ds2.keys())[0]
    datasets.append(ds2[first_split])
    print(f"    ✅ {len(ds2[first_split])} samples")
except Exception as e:
    print(f"    ⚠️  Skipped: {e}")

if datasets:
    dataset = concatenate_datasets(datasets)
    print(f"\n✅ Combined: {len(dataset)} total samples\n")
else:
    print("❌ No datasets loaded!")
    sys.exit(1)

# ===== SETUP MODEL & PROCESSOR =====
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
print("📦 Loading model and processor...")

try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "./checkpoint-250",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("✅ Loaded checkpoint-250")
except:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("✅ Loaded base model")

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# ===== APPLY ENHANCED LoRA =====
print("\n🔧 Applying Enhanced LoRA (Phase 2C):")
print(f"  Rank:       64 (↑ from 32)")
print(f"  Alpha:      128 (↑ from 64)\n")

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, lora_config)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"📊 Parameters: {trainable_params:,} / {total_params:,} trainable ({100*trainable_params/total_params:.2f}%)\n")

# ===== CUSTOM DATASET WRAPPER =====
class OCRDataset(torch.utils.data.Dataset):
    """Minimal dataset wrapper"""
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
        self.cache = {}
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        
        # Skip invalid examples
        if example.get("image") is None or example.get("text") is None:
            return self[(idx + 1) % len(self.dataset)]
        
        try:
            image = example["image"]
            if not isinstance(image, type(None)):
                if hasattr(image, 'convert'):
                    image = image.convert("RGB")
            text = f"<image>\n{example['text']}"
            
            # Process to get model inputs
            inputs = self.processor(text=[text], images=[image], return_tensors="pt")
            
            # Flatten batch dimension
            flat_inputs = {}
            for k, v in inputs.items():
                if v is not None:
                    flat_inputs[k] = v.squeeze(0)
            
            return flat_inputs
        except Exception as e:
            # Skip problematic examples
            return self[(idx + 1) % len(self.dataset)]


# Wrap dataset
ocr_dataset = OCRDataset(dataset, processor)

# ===== TRAINING CONFIG =====
print("⚙️  Training Configuration:")
print(f"  Epochs:     1")
print(f"  Batch:      1 (effective: 4)")
print(f"  LR:         1e-4")
print(f"  Device:     GPU (A100 auto-mapped)\n")

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
    remove_unused_columns=False,
)

# ===== SIMPLE COLLATOR =====
def collate_fn(batch):
    """Collate processor outputs"""
    if not batch:
        return {}
    
    # Get keys from first example
    keys = batch[0].keys()
    result = {}
    
    for key in keys:
        tensors = [example[key] for example in batch if key in example]
        if tensors and tensors[0] is not None:
            try:
                result[key] = torch.stack(tensors)
            except:
                result[key] = torch.cat(tensors, dim=0)
    
    return result

# ===== START TRAINING =====
print("🔄 Initializing Trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ocr_dataset,
    data_collator=collate_fn,
)

print("🟢 Starting Training...\n")
print("="*70)

try:
    trainer.train()
    print("\n" + "="*70)
    print("✅ Phase 2C Training Complete!")
    print("="*70)
    print("✅ Model: checkpoint-300-phase2c (target: 20% CER)")
    print("📊 Next: Phase 3 will auto-launch for final 5% improvement\n")
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
