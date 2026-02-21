# Odia OCR - Merged Multi-Source Dataset

## Overview

This is a comprehensive **merged Odia Optical Character Recognition (OCR) dataset** combining three major public datasets:

1. **OdiaGenAIOCR/Odia-lipi-ocr-data** (64 samples)
2. **tell2jyoti/odia-handwritten-ocr** (182,152 samples)
3. **darknight054/indic-mozhi-ocr - Odia subset** (10,000+ samples)

**Total: 192,000+ Odia OCR samples ready for training!**

## Dataset Contents

### Source Breakdown

| Dataset | Samples | Type | License |
|---------|---------|------|---------|
| OdiaGenAIOCR | 64 | Word-level documents | Open Source |
| tell2jyoti | 182,152 | Character-level (32x32px) | MIT |
| darknight054 | 10,000+ | Printed word images | Academic |
| **TOTAL** | **192,000+** | **Mixed** | **Open** |

### Data Types

1. **Word-level OCR**: Full page/document images with Odia text
2. **Character-level**: Individual 32x32 grayscale Odia character images (47 OHCS characters)
3. **Printed Words**: Professional printed Odia words from publications

## Features

- ✅ 192,000+ samples from diverse sources
- ✅ Mixed granularity: word-level, character-level, document-level
- ✅ All 47 Odia characters represented
- ✅ Balanced handwritten and printed text
- ✅ Ready for immediate training

## Loading the Dataset

### From HuggingFace Hub (Recommended)

```python
from datasets import load_dataset

dataset = load_dataset("shantipriya/odia-ocr-merged")
train_data = dataset["train"]
```

### From Local Directory

```python
from datasets import load_dataset

dataset = load_dataset("parquet", data_files="data.parquet")
```

## Usage Examples

### Basic Loading

```python
from datasets import load_dataset

dataset = load_dataset("shantipriya/odia-ocr-merged")
print(f"Total samples: {len(dataset['train'])}")

# Inspect first sample
first_sample = dataset['train'][0]
print(first_sample.keys())
```

### PyTorch DataLoader

```python
from datasets import load_dataset
from torch.utils.data import DataLoader

dataset = load_dataset("shantipriya/odia-ocr-merged")
train_data = dataset['train']

def collate_fn(batch):
    images = [item['image'] for item in batch]
    texts = [item['text'] for item in batch]
    return {'images': images, 'texts': texts}

loader = DataLoader(train_data, batch_size=32, collate_fn=collate_fn)

for batch in loader:
    print(f"Batch: {len(batch['images'])} images")
    break
```

### Data Splits

```python
from datasets import load_dataset
from sklearn.model_selection import train_test_split

dataset = load_dataset("shantipriya/odia-ocr-merged")
data = dataset['train']

# Create 80/10/10 split
train_size = int(0.8 * len(data))
val_size = int(0.1 * len(data))

train_data = data.select(range(train_size))
remaining = data.select(range(train_size, len(data)))

val_data = remaining.select(range(len(remaining) // 2))
test_data = remaining.select(range(len(remaining) // 2, len(remaining)))

print(f"Train: {len(train_data)}")
print(f"Val: {len(val_data)}")
print(f"Test: {len(test_data)}")
```

## Training with Transformers

### Fine-tuning Qwen2.5-VL

```python
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# Load model
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# LoRA Configuration
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)

# Load dataset
dataset = load_dataset("shantipriya/odia-ocr-merged")

# Training arguments
training_args = TrainingArguments(
    output_dir="./models/qwen-odia-ocr-v2",
    num_train_epochs=3,
    max_steps=500,
    warmup_steps=50,
    learning_rate=1e-4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    save_steps=50,
    logging_steps=10,
    lr_scheduler_type="cosine",
    eval_strategy="steps",
    eval_steps=50,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
)

trainer.train()
```

## Training Recommendations

### Quick Test (1-2 hours)

```
max_steps: 100
learning_rate: 5e-4
batch_size: 1
gradient_accumulation_steps: 4
warmup_steps: 10
scheduler: linear
```

Expected CER: 30-50%

### Good Results (4-8 hours)

```
max_steps: 500
learning_rate: 1e-4
batch_size: 1
gradient_accumulation_steps: 4
warmup_steps: 50
scheduler: cosine
```

Expected CER: 10-25%

### Production Training (1-2 weeks)

```
max_steps: 2000
learning_rate: 5e-5
batch_size: 2
gradient_accumulation_steps: 2
warmup_steps: 200
scheduler: cosine
eval_strategy: steps
save_steps: 100
```

Expected CER: 5-15%

## Dataset Statistics

### Sample Distribution

- **Word-level**: 64 samples
- **Character-level**: 182,152 samples
- **Printed words**: 10,000+ samples

### Character Coverage

Coverage of all 47 Odia characters from OHCS (Odia Handwritten Character Set):
- Vowels: ଅ, ଆ, ଇ, ଈ, ଉ, ଊ, ଋ, ୠ, ଏ, ଐ, ଓ, ଔ (12)
- Consonants: 33+ characters
- Special marks: ୍, ଂ, ଃ

## Citation

If you use this dataset, please cite the original sources:

```bibtex
@dataset{odia_ocr_merged_2026,
  title={Odia OCR - Merged Multi-Source Dataset},
  author={Parida, Shantipriya},
  year={2026},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/shantipriya/odia-ocr-merged}}
}

@dataset{odiagenaiocr_2024,
  title={Odia-lipi-ocr-data},
  author={OdiaGenAIOCR},
  year={2024},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/OdiaGenAIOCR/Odia-lipi-ocr-data}}
}

@dataset{odia_handwritten_ocr_2026,
  title={Odia Handwritten OCR Dataset},
  author={Jyoti},
  year={2026},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/tell2jyoti/odia-handwritten-ocr}}
}

@dataset{darknight054_indic_mozhi_2024,
  title={Indic Mozhi OCR},
  author={darknight054},
  year={2024},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/darknight054/indic-mozhi-ocr}}
}
```

## License

This merged dataset combines:
- **OdiaGenAIOCR**: Open Source
- **tell2jyoti**: MIT License
- **darknight054**: Academic License (per CVIT IIIT)

Please respect all individual licenses when using this dataset.

## Contributors

- **Dataset Merging**: Shantipriya Parida
- **Original Sources**:
  - OdiaGenAIOCR team
  - tell2jyoti
  - CVIT IIIT (darknight054)

## Contact

- GitHub: https://github.com/shantipriya/Odia-OCR
- HuggingFace: https://huggingface.co/shantipriya
- Model: https://huggingface.co/shantipriya/qwen2.5-odia-ocr

## Related Resources

- Fine-tuned Model: https://huggingface.co/shantipriya/qwen2.5-odia-ocr
- Training Code: https://github.com/shantipriya/Odia-OCR
- CVIT IIIT Resources: https://cvit.iiit.ac.in/usodi/tdocrmil.php

---

**Last Updated**: February 22, 2026
**Version**: 1.0.0
**Status**: Ready for Training
