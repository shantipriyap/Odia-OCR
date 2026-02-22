# 🚀 Odia OCR - How to Improve Further

**Current Status**: Phase 2A Complete (32% CER)  
**Timeline to Production**: 6-12 weeks to reach <10% CER

---

## Quick Summary: 5-Phase Improvement Plan

```
32% CER → 26% (Phase 2B) → 20% (Phase 2C) → 15% (Phase 3) → 8% (Phase 4) → 5% (Phase 5)
  [Week 1]     [Week 2]     [3 days]       [1 month]    [2 months]
```

---

## Phase 2B: Quick Wins (26% CER Target) - Start This Week

### B1: Odia Spell Correction (3-4% improvement)
```python
# Install symspellpy
pip install symspellpy

# Create Odia dictionary with common words
from symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=2)

# Top 100 Odia OCR error corrections
CORRECTIONS = {
    "ସି": "ସ ",  # Common confusion
    "ଚିଜ": "ଚିଜ୍ଞ",  # Character cluster
    "ଧାନ": "ଧାନ୍ଯ",  # Diacritical
}

def fix_odia_text(text):
    words = text.split()
    corrected = []
    for word in words:
        if word in CORRECTIONS:
            corrected.append(CORRECTIONS[word])
        else:
            suggestions = sym_spell.lookup(word, Verbosity.CLOSEST)
            corrected.append(suggestions[0].term if suggestions else word)
    return " ".join(corrected)
```

### B2: Language Model Reranking (2-4% improvement)
```python
# Use Odia GPT-2 to score predictions
from transformers import AutoModelForCausalLM, AutoTokenizer

odia_lm = AutoModelForCausalLM.from_pretrained("sarvamai/odia-gpt2")
tokenizer = AutoTokenizer.from_pretrained("sarvamai/odia-gpt2")

def score_prediction(text):
    """Score text likelihood using Odia LM"""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        loss = odia_lm(**inputs, labels=inputs["input_ids"]).loss
    return -loss.item()  # Higher = more likely

# Generate multiple candidates and score
candidates = beam_search(model, image, num_beams=5)
best = max(candidates, key=score_prediction)
```

### B3: Diacritical Mark Consistency (1-2% improvement)
```python
# Enforce consistency for repeated characters
def fix_diacriticals(text):
    """Fix inconsistent diacritical marks"""
    # If 'ଗୁ' appears as both 'ଗୁ' and 'ଗ୍ଯୁ', standardize
    replacements = {
        'ଇ': 'ି',  # Incorrect form → correct
        'ଏ': 'େ',  # Incorrect form → correct
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text
```

**Expected Result**: 32% → 26% CER (1 week effort)

---

## Phase 2C: Model Enhancement (20% CER Target) - Week 2

### C1: Increase LoRA Rank (2-3% improvement)
```python
from peft import LoraConfig, get_peft_model

# Current: r=32, New: r=64
peft_config = LoraConfig(
    r=64,  # ← CHANGE FROM 32
    lora_alpha=128,  # Should be 2*r
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
# Model size: 28MB → 56MB (still very compact)
```

### C2: Add More Attention Layers for Fine-tuning (2% improvement)
```python
# Expand target modules for better adaptation
peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",  # Visual attention
        "fc1", "fc2",  # Feed-forward layers
    ],
    task_type="CAUSAL_LM",
)
```

### C3: Data Augmentation (2-3% improvement)
```python
import albumentations as A

transform = A.Compose([
    A.Rotate(limit=15, p=0.5),  # Natural rotation
    A.Affine(scale=(0.9, 1.1), p=0.3),  # Scaling
    A.GaussNoise(p=0.2),  # Noise
    A.GaussianBlur(blur_limit=3, p=0.2),  # Blur (like poor scan)
    A.Perspective(scale=(0.05, 0.1), p=0.3),  # Perspective
])

# Apply during training batch preprocessing
def augment_batch(batch):
    images = [transform(image=img)["image"] for img in batch]
    return images
```

**Expected Result**: 26% → 20% CER (1 week effort, with augmentation training)

---

## Phase 3: Complete Training (15% CER Target) - Week 3-4

### Resume and Complete Training to 500 Steps
```bash
# Modify training_ocr_qwen.py
# Change: max_steps = 500  (from 250)
# Change: save_steps = 50
# Add: evaluation_steps = 50

python training_ocr_qwen.py
# Estimated time: 3-4 days on RTX A6000
```

**Key Changes**:
```python
training_args = TrainingArguments(
    output_dir="./qwen_odia_ocr_improved_v3",
    num_train_epochs=1,
    max_steps=500,  # ← INCREASE
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-4,
    
    # NEW: Better scheduling
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    
    # NEW: Gradient accumulation for larger effective batch
    gradient_accumulation_steps=2,
    
    # NEW: Early stopping
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)
```

**Expected Result**: 20% → 15% CER (3-4 days GPU time)

---

## Phase 4: Advanced Optimization (8% CER Target) - Month 2

### A4.1: Knowledge Distillation
```python
# Use larger teacher model to improve smaller student
from torch.nn import KLDivLoss

def distillation_loss(student_logits, teacher_logits, temp=3.0):
    """Kullback-Leibler divergence loss"""
    soft_targets = torch.nn.functional.softmax(teacher_logits / temp, dim=-1)
    soft_pred = torch.nn.functional.log_softmax(student_logits / temp, dim=-1)
    return KLDivLoss(reduction='batchmean')(soft_pred, soft_targets)

# Use teacher (larger model) predictions to regularize student
```

### A4.2: Quantization for Deployment (maintain quality, 4x faster)
```python
# INT8 quantization
from transformers import AutoModelForVision2Seq
import torch

model = AutoModelForVision2Seq.from_pretrained("...")
model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
# After quantization: 500MB → 125MB, 4x faster inference
```

### A4.3: Ensemble with Diverse Methods
```python
def ensemble_inference(image):
    """Combine multiple approaches"""
    
    # Method 1: Greedy (fast baseline)
    pred1 = model.generate(inputs, max_new_tokens=256)
    
    # Method 2: Beam search (higher quality)
    pred2 = model.generate(inputs, max_new_tokens=256, num_beams=5)
    
    # Method 3: Temperature sampling
    pred3 = model.generate(inputs, max_new_tokens=256, temperature=0.8)
    
    # Vote for best
    predictions = [pred1, pred2, pred3]
    best = majority_vote(predictions)
    return best
```

**Expected Result**: 15% → 8% CER (3-4 weeks effort)

---

## Phase 5: Domain Specialization (5% CER Target) - Month 3

### D5.1: Handwritten Text Adaptation
```python
# Fine-tune specifically on handwritten samples
handwritten_loader = DataLoader(handwritten_dataset, batch_size=4)

for epoch in range(1):
    for batch in handwritten_loader:
        # Transfer learning from main model
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
```

### D5.2: Document Type Classification
```python
# Detect document type and use specialized model
def classify_document(image):
    """Classify: book, newspaper, handwritten, historical"""
    classifier = load_document_classifier()
    doc_type = classifier(image)
    
    if doc_type == "handwritten":
        model = load_model("odia_ocr_handwritten")
    elif doc_type == "historical":
        model = load_model("odia_ocr_historical")
    else:
        model = load_model("odia_ocr_general")
    
    return model.generate(image)
```

**Expected Result**: 8% → 5% CER (6-8 weeks for all specializations)

---

## olmOCR-Bench Evaluation Strategy

### Integration with ollmOCR-Bench

```bash
# 1. Install olmOCR
pip install olmocr[bench]

# 2. Download benchmark (or create Odia subset)
huggingface-cli download allenai/olmOCR-bench --local-dir ./olmocr_bench

# 3. Generate predictions
python generate_predictions.py --model=shantipriya/qwen2.5-odia-ocr

# 4. Evaluate
python run_eval.py -d olmocr_bench/bench_data
```

### Create Odia-Specific Benchmark Categories

**Recommended Test Suite** (1,650 samples):
- **Odia Script Basics** (500): Character recognition, diacriticals
- **Diacritical Marks** (300): Complex mark combinations  
- **Literature** (200): Classic texts, poetry
- **Multi-column** (150): Newspaper layout
- **Handwritten** (100): Handwriting samples
- **Tables** (150): Data extraction
- **Math/Science** (100): Equations with Odia text

---

## Priority Checklist

### Week 1 (Start Today)
- [ ] Create Odia spell-correction dictionary (top 500 words)
- [ ] Implement spell-correction post-processing
- [ ] Run basic benchmark: current CER vs corrected CER
- [ ] Document improvements

### Week 2
- [ ] Increase LoRA rank to 64
- [ ] Create data augmentation pipeline
- [ ] Train for 50 more steps with augmentation
- [ ] Measure CER improvement

### Week 3-4
- [ ] Resume training to 500 total steps
- [ ] Monitor validation loss
- [ ] Create evaluation report

### Month 2
- [ ] Implement knowledge distillation
- [ ] Quantize model for deployment
- [ ] Test ensemble inference

### Month 3+
- [ ] Domain specialization models
- [ ] Full olmOCR-Bench evaluation
- [ ] Production deployment

---

## Expected Performance Trajectory

| Phase | CER | WER | Exact Match | Effort | Time |
|-------|-----|-----|-------------|--------|------|
| Current (2A) | 32% | 68% | 24% | - | - |
| 2B (Spell Corr) | 26% | 60% | 32% | Medium | 1 wk |
| 2C (Model Enhance) | 20% | 52% | 40% | High | 1 wk |
| 3 (Full Train) | 15% | 45% | 50% | Medium | 4 days |
| 4 (Optimization) | 8% | 30% | 70% | High | 4 wks |
| 5 (Specialization) | 5% | 20% | 85% | High | 8 wks |

---

## Recommended Tech Stack

**For Post-Processing**:
- `symspellpy` - Spell correction
- `sarvamai/odia-gpt2` - Language model scoring

**For Training Enhancements**:
- `albumentations` - Data augmentation
- `transformers` (latest) - Training utilities
- `peft` - LoRA management

**For Deployment**:
- `torch.quantization` - Model compression
- `triton` - Efficient inference
- `ray` - Distributed processing

**For Evaluation**:
- `olmocr` - Official benchmark
- `jiwer` - CER/WER metrics
- `evaluate` - HF metrics library

---

## Getting Started Now

```bash
# Step 1: Create spell correction
python -c "
import json
odia_words = {
    'ସଠିକ': 'ସୁଠିକ',  # Common errors
    # Add 100 more...
}
with open('odia_corrections.json', 'w') as f:
    json.dump(odia_words, f, ensure_ascii=False)
"

# Step 2: Run evaluation with corrections
python post_process_predictions.py --model=shantipriya/qwen2.5-odia-ocr

# Step 3: Benchmark improvement
python evaluate_improvements.py
```

---

## Questions to Answer First

1. **Do you have Odia spell-checker dictionary?**  
   - Create one from training data (most common words)

2. **Can you get more Odia GPT-2 training data?**  
   - Use Wikipedia Odia, books, online content

3. **What's the target use case?**  
   - Historical texts? Newspapers? Handwritten? → Affects optimization priority

4. **GPU availability for training?**  
   - RTX A6000 (current): 3-4 days for 250→500 steps
   - RTX 4090: 1-2 days
   - Multiple GPUs: Proportional speedup

5. **Production latency requirement?**  
   - <1 sec? → Need quantization + optimization
   - 5-10 sec ok? → Use ensemble voting

---

**Next Step**: Pick Phase 2B items, start with spell correction, benchmark improvements!
