from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
import torch
import os

# 1) Load the dataset
dataset = load_dataset("OdiaGenAIOCR/Odia-lipi-ocr-data")

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
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
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
    learning_rate=1e-5,
    max_steps=100,
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    fp16=False,  # Disable FP16 to reduce memory footprint
    remove_unused_columns=False,
    dataloader_num_workers=0,
    optim="adamw_torch",
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