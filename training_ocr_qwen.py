from datasets import load_dataset, concatenate_datasets
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
import torch
import os
import io

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
    handwritten_root = None
    
    # Primary dataset
    print("📥 Loading primary dataset: OdiaGenAIOCR/Odia-lipi-ocr-data")
    ds1 = load_dataset("OdiaGenAIOCR/Odia-lipi-ocr-data")
    datasets.append(ds1["train"])
    print(f"   ✅ Loaded: {len(ds1['train'])} samples\n")
    
    # Optional: Add handwritten dataset
    if use_multiple:
        try:
            print("Loading secondary dataset: tell2jyoti/odia-handwritten-ocr")
            ds2 = load_dataset("tell2jyoti/odia-handwritten-ocr")
            first_split = list(ds2.keys())[0]
            datasets.append(ds2[first_split])
            handwritten_root = snapshot_download(
                "tell2jyoti/odia-handwritten-ocr",
                repo_type="dataset",
                local_dir="/root/odia_ocr/datasets/odia-handwritten-ocr",
                local_dir_use_symlinks=False,
            )
            print(f"Loaded: {len(ds2[first_split])} samples (split: {first_split})\n")
        except Exception as e:
            print(f"Could not load handwritten dataset: {e}\n")
    
    # Combine all datasets
    if len(datasets) > 1:
        combined = concatenate_datasets(datasets)
        print(f"✅ COMBINED DATASET: {len(combined)} total samples")
        print(f"   • OdiaGenAIOCR: {len(datasets[0])} samples")
        if len(datasets) > 1:
            print(f"   • tell2jyoti: {len(datasets[1])} samples")
        print()
        return combined, handwritten_root
    else:
        return datasets[0], handwritten_root

# Load dataset (set use_multiple=True to include handwritten data)
dataset, handwritten_root = load_odia_datasets(use_multiple=True)

def resolve_image_from_example(example):
    if example.get("image") is not None:
        image = example["image"]
        while isinstance(image, list):
            image = image[0] if image else None
        if isinstance(image, dict) and "bytes" in image:
            image = Image.open(io.BytesIO(image["bytes"]))
    elif "image_path" in example:
        image_path = example["image_path"]
        if handwritten_root and not os.path.isabs(image_path):
            image_path = os.path.join(handwritten_root, image_path)
        if not os.path.exists(image_path):
            return None, None
        with Image.open(image_path) as img:
            image = img.copy()
    else:
        return None, None

    text = example.get("text") or example.get("character")
    while isinstance(text, list):
        text = text[0] if text else None
    if text is None:
        return None, None
    return image, text

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
# PHASE 2C ENHANCEMENT: Increased LoRA capacity for improved learning
lora_config = LoraConfig(
    r=64,                              # PHASE 2C: Increased from 32 to 64
    lora_alpha=128,                    # PHASE 2C: Increased from 64 to 128
    target_modules=["q_proj", "v_proj"],  # Focused: remove k_proj, o_proj for stability
    lora_dropout=0.05,                 # Reduced: lower dropout for better signal
)

# 4) Filter and normalize samples (supports image and image_path datasets)
print("Filtering samples with valid images...")

def has_valid_sample(example):
    text = example.get("text") or example.get("character")
    while isinstance(text, list):
        text = text[0] if text else None
    if text is None:
        return False
    image = example.get("image")
    while isinstance(image, list):
        image = image[0] if image else None
    if image is not None:
        return True
    if "image_path" in example:
        image_path = example["image_path"]
        if handwritten_root and not os.path.isabs(image_path):
            image_path = os.path.join(handwritten_root, image_path)
        return os.path.exists(image_path)
    return False

dataset = dataset.filter(has_valid_sample)
print(f"After filtering: {len(dataset)} samples with valid images\n")

# Avoid eager image loading; resolve per sample via a wrapper dataset
class OCRDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        image, text = resolve_image_from_example(example)
        while isinstance(image, list):
            image = image[0] if image else None
        if image is None:
            raise ValueError("Encountered sample without a valid image")
        image = image.convert("RGB")
        return {
            "image": image,
            "text": text,
        }

train_dataset = OCRDataset(dataset)

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
    # Note: Removed eval_steps and evaluation_strategy - not supported in this Transformers version
    lr_scheduler_type="cosine",        # IMPROVED: cosine decay instead of linear
    fp16=False,
    remove_unused_columns=False,
    dataloader_num_workers=0,
    optim="adamw_torch",
    # Note: Removed load_best_model_at_end and metric_for_best_model (require evaluation_strategy)
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
        data_collator=data_collator,
    )

    trainer.train()