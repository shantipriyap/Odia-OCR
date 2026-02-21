---
license: mit
tags:
  - odia
  - ocr
  - vision-language
  - qwen2.5-vl
  - lora
  - fine-tuning
language:
  - or
datasets:
  - shantipriya/odia-ocr-merged
---

# Qwen2.5-VL Fine-tuned for Odia OCR

This is a fine-tuned version of [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) trained on the [shantipriya/odia-ocr-merged](https://huggingface.co/datasets/shantipriya/odia-ocr-merged) dataset.

## 📊 Training Details

- **Base Model**: Qwen/Qwen2.5-VL-3B-Instruct
- **Dataset**: shantipriya/odia-ocr-merged (145,781 samples)
- **Method**: LoRA fine-tuning
  - Rank: 32
  - Alpha: 64
  - Target Modules: q_proj, v_proj
- **Training Config**:
  - Steps: 500
  - Learning Rate: 1e-4
  - Scheduler: Cosine with warmup (50 steps)
  - Batch Size: 4 (1 per device × 4 gradient accumulation)
- **Training Data Split**: 80/10/10 (train/val/test)

## 🎯 Performance

**Expected Improvement**:
- Baseline CER: 100%
- Phase 1 Target: 30-50% CER
- Expected Improvement: 10-40x

## 📝 Dataset Details

The merged dataset combines three Odia OCR sources:

1. **OdiaGenAIOCR/Odia-lipi-ocr-data**: 64 word-level samples
2. **tell2jyoti/odia-handwritten-ocr**: 145,717 character-level samples (32×32px, 47 OHCS characters)
3. **darknight054/indic-mozhi-ocr**: 10,000+ printed word samples

**Total**: 145,781 unique Odia text-image pairs

## 🚀 Usage

```python
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import requests

# Load model and processor
model_id = "shantipriya/qwen2.5-odia-ocr-v2"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype="auto",
    trust_remote_code=True,
    device_map="auto"
)

# Load image
url = "https://example.com/odia_text.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Process and generate
inputs = processor(image, text="Recognize Odia text:", return_tensors="pt")
output_ids = model.generate(**inputs)
text = processor.decode(output_ids[0])
print(text)
```

## 📂 Checkpoints

This model contains checkpoints saved at every 50 training steps:
- `checkpoints/checkpoint-50` (10%)
- `checkpoints/checkpoint-100` (20%)
- `checkpoints/checkpoint-150` (30%)
- ... and more

See the model repository for all available checkpoints.

## 🔗 Related Resources

- **Merged Dataset**: https://huggingface.co/datasets/shantipriya/odia-ocr-merged
- **Base Model**: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
- **PEFT**: https://huggingface.co/docs/peft/

## 📄 License

MIT

## 👏 Acknowledgments

- **Dataset Sources**: OdiaGenAIOCR, tell2jyoti, darknight054
- **Framework**: Hugging Face Transformers, PEFT
- **Base Model**: Qwen Team

---

*Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
