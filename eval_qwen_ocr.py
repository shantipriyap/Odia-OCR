import unicodedata
import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from jiwer import cer, wer
from tqdm import tqdm


# --------------------
# CONFIG
# --------------------
MODEL_PATH = "./qwen_ocr_finetuned"   # or HF repo path
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_NAME = "OdiaGenAIOCR/Odia-lipi-ocr-data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 256


# --------------------
# LOAD MODEL & PROCESSOR
# --------------------
# Prefer processor from the fine-tuned folder; fall back to the base model
try:
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
except Exception:
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

device_map = {"": "cuda:0"} if torch.cuda.is_available() else None
try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map=device_map,
    )
except Exception:
    # MODEL_PATH may contain only PEFT adapters. Load base model then apply adapters.
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base, MODEL_PATH, device_map=device_map)

model.eval()


# --------------------
# LOAD DATASET
# --------------------
dataset = load_dataset(DATASET_NAME)["train"]


# --------------------
# NORMALIZATION (VERY IMPORTANT FOR ODIA)
# --------------------
def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    return text


# --------------------
# OCR INFERENCE
# --------------------
def ocr_predict(image):
    inputs = processor(
        images=image.convert("RGB"),
        return_tensors="pt"
    )

    # move inputs to model device if model is on CUDA
    if device_map:
        target_device = device_map[""]
    else:
        target_device = DEVICE

    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS
        )

    pred = processor.decode(
        output_ids[0],
        skip_special_tokens=True
    )
    return normalize_text(pred)


# --------------------
# RUN EVALUATION
# --------------------
predictions = []
references = []

for sample in tqdm(dataset, desc="Evaluating OCR"):
    gt_text = normalize_text(sample["text"])
    pred_text = ocr_predict(sample["image"])

    predictions.append(pred_text)
    references.append(gt_text)

    print("GT :", gt_text)
    print("PR :", pred_text)
    print("-" * 40)


# --------------------
# METRICS
# --------------------
cer_score = cer(references, predictions)
wer_score = wer(references, predictions)

print("\n===== OCR Evaluation Results =====")
print(f"CER : {cer_score:.4f}")
print(f"WER : {wer_score:.4f}")