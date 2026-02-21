from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
import torch
import os

# ============================================================================
# 1) LOAD MULTIPLE ODIA OCR DATASETS
# ============================================================================

def load_odia_datasets(use_multiple=False):
    """
    Load Odia OCR datasets
    
    Args:
        use_multiple: If True, combine multiple sources including handwritten dataset
    
    Returns:
        Loaded dataset
    """
    
    datasets = []
    
    # Primary dataset
    print("📥 Loading primary dataset: OdiaGenAIOCR/Odia-lipi-ocr-data")
    ds1 = load_dataset("OdiaGenAIOCR/Odia-lipi-ocr-data")
    datasets.append(ds1["train"])
    print(f"   ✅ Loaded: {len(ds1['train'])} samples\n")
    
    # Optional: Add handwritten dataset
    if use_multiple:
        try:
            print("📥 Loading secondary dataset: tell2jyoti/odia-handwritten-ocr")
            ds2 = load_dataset("tell2jyoti/odia-handwritten-ocr")
            first_split = list(ds2.keys())[0]
            datasets.append(ds2[first_split])
            print(f"   ✅ Loaded: {len(ds2[first_split])} samples (split: {first_split})\n")
        except Exception as e:
            print(f"   ⚠️  Could not load handwritten dataset: {e}\n")
    
    # Combine all datasets
    if len(datasets) > 1:
        combined = concatenate_datasets(datasets)
        print(f"✅ COMBINED DATASET: {len(combined)} total samples")
        print(f"   • OdiaGenAIOCR: {len(datasets[0])} samples")
        if len(datasets) > 1:
            print(f"   • tell2jyoti: {len(datasets[1])} samples")
        print()
        return combined
    else:
        return datasets[0]

# Load dataset (set use_multiple=True to include handwritten data)
dataset = load_odia_datasets(use_multiple=True)

# 2) Load Qwen 2.5 VL processor (model instantiated under __main__)
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# Find or add an image placeholder token so image features align with text tokens
def find_image_token(processor):
    tokenizer = processor.tokenizer
    candidates = ["<image>", "<Image>", "<img>", "<image_patch>", "<image_token>"]
    for c in candidates:
        try:
            tid = tokenizer.convert_tokens_to_ids(c)
            if tid is not None and hasattr(tokenizer, "unk_token_id") and tid != tokenizer.unk_token_id:
                return c
        except Exception:
            pass
    for t in getattr(tokenizer, "all_special_tokens", []):
        if "image" in t.lower() or "img" in t.lower():
            return t
    return "<image>"

image_token = find_image_token(processor)
if image_token not in processor.tokenizer.get_vocab():
    processor.tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})

# 3) Add QLoRA (low-rank adapters) so the base stays frozen
lora_config = LoraConfig(
    r=32,                              # Increased: better representation
    lora_alpha=64,                     # Increased: 2x scaling
    target_modules=["q_proj", "v_proj"],  # Focused: remove k_proj, o_proj for stability
    lora_dropout=0.05,                 # Reduced: lower dropout for better signal
)

# 4) Preprocess examples - process individually to avoid tensor batching issues
def preprocess_function(example):
    image = example["image"].convert("RGB")
    # Return raw image and text - processor will be called in data_collator
    return {
        "image": image,
        "text": example["text"]
    }

# Process without batching
train_dataset = dataset["train"].map(preprocess_function, batched=False)
eval_dataset = dataset["train"].map(preprocess_function, batched=False)  # No val split yet

# 4b) Custom data collator to handle processor at batch time
class QwenOCRDataCollator:
    def __init__(self, processor):
        self.processor = processor
        # cache image token
        try:
            self.image_token = find_image_token(processor)
        except Exception:
            self.image_token = "<image>"
        # We insert a single image placeholder token per image; the
        # processor will expand that token into the correct number of
        # placeholder tokens based on the image grid.
        self.num_image_tokens = 1
    
    def __call__(self, batch):
        images = [example["image"] for example in batch]
        texts = [example["text"] for example in batch]
        # Insert a single image placeholder token at the start of each text so
        # the processor can expand it to the needed number of visual tokens.
        texts = [f"{self.image_token} {t}" for t in texts]

        # Process batch of images with texts
        inputs = self.processor(
            images,
            text=texts,
            padding=True,
            truncation=False,
            return_tensors="pt"
        )
        
        # Set labels
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

data_collator = QwenOCRDataCollator(processor)

# 5) Training arguments
training_args = TrainingArguments(
    output_dir="./qwen_ocr_finetuned",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=500,                     # Increased: from 100 to 500 steps
    warmup_steps=50,                   # NEW: 10% warmup for better stability
    learning_rate=1e-4,                # Reduced: from 2e-4 for better convergence
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    eval_steps=50,                     # NEW: evaluate during training
    evaluation_strategy="steps",       # NEW: track eval metrics
    lr_scheduler_type="cosine",        # IMPROVED: cosine decay instead of linear
    fp16=False,
    remove_unused_columns=False,
    dataloader_num_workers=0,
    optim="adamw_torch",
    load_best_model_at_end=True,       # NEW: keep best checkpoint
    metric_for_best_model="eval_loss",  # NEW: track by eval loss
)

# 6) Trainer
trainer = None

# quick verification utility (import-safe)
def quick_forward_check(n=2):
    batch = [train_dataset[i] for i in range(n)]
    inputs = data_collator(batch)
    # quick_forward_check will only run a model forward if `model` is available
    if 'model' in globals() and globals()['model'] is not None:
        return globals()['model'](**inputs)
    return inputs


if __name__ == "__main__":
    # Instantiate the model and wrap with PEFT, then create Trainer and start fine-tuning
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="auto",
    )

    # If we added special tokens earlier, resize embeddings now that model is available
    model.resize_token_embeddings(len(processor.tokenizer))

    # Apply LoRA adapters
    model = get_peft_model(model, lora_config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()