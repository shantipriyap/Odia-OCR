# 🎯 Model Accuracy Improvement Guide - Odia OCR

## Executive Summary

Current Status: **100% CER** (100% error rate - model not extracting text)  
Target: **< 20% CER** for production use

**Why is accuracy low?**
- Only 100 training steps (need 500-1000+)
- Only 64 training samples (need 500+)
- Model hasn't learned Odia script patterns yet
- Proof-of-concept stage, not production-ready

---

## 📈 Improvement Strategy (Prioritized)

### Phase 1: Quick Wins (1-2 hours implementation)

#### 1.1 **Increase Training Steps** ⭐ HIGHEST IMPACT
```python
# Current: 100 steps
# Recommended: 500-1000 steps
# Change in training_ocr_qwen.py:

training_args = TrainingArguments(
    ...
    max_steps=500,        # Increase from 100
    save_steps=50,        # Save every 50 steps (10 checkpoints)
    eval_steps=50,        # Evaluate every 50 steps
    evaluation_strategy="steps",  # Enable evaluation
    ...
)
```

**Expected Impact:** CER reduction from 100% → ~30-50%  
**Time Required:** ~5-10 minutes training  
**Cost:** Minimal (using existing GPU)

#### 1.2 **Add Warmup Period**
```python
training_args = TrainingArguments(
    ...
    max_steps=500,
    warmup_steps=50,      # 5-10% of total steps
    warmup_ratio=0.1,     # Alternative: 10% warmup
    ...
)
```

**Why:** Helps model settle into good parameter space  
**Expected Impact:** Faster convergence, better stability  

#### 1.3 **Better Learning Rate Schedule**
```python
training_args = TrainingArguments(
    ...
    learning_rate=1e-4,              # Lower from 2e-4
    lr_scheduler_type="cosine",      # Better than linear
    # or
    lr_scheduler_type="polynomial",
    ...
)
```

**Expected Impact:** Better convergence curve, 5-10% accuracy improvement

---

### Phase 2: Data & Hyperparameter Optimization (1-2 days)

#### 2.1 **Collect More Training Data** ⭐ HIGHEST IMPACT
```
Current: 64 samples
Target: 500-1000 samples
How: 
  • Digitize more Odia documents
  • Find public domain Odia books
  • Use OCR synthetic data generation
  • Community contributions
```

**Expected Impact:** CER reduction ~30-50%  
**Implementation:**
```python
# Update dataset loading in training_ocr_qwen.py
dataset = load_dataset(
    "OdiaGenAIOCR/Odia-lipi-ocr-data",
    split="train"
)
# After augmentation should grow to 500+ samples
```

#### 2.2 **Data Augmentation**
```python
# Add to dataset processing pipeline
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance

def augment_image(image):
    """Augment training images for better generalization"""
    
    # 1. Random rotation (±5 degrees)
    angle = random.uniform(-5, 5)
    image = image.rotate(angle)
    
    # 2. Random brightness adjustment
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))
    
    # 3. Random contrast adjustment
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))
    
    # 4. Random sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))
    
    return image
```

**Expected Impact:** 10-20% accuracy improvement  
**Code Integration:** Apply in dataset preprocessing

#### 2.3 **Batch Size Optimization**
```python
# Current: batch_size=1, gradient_accumulation=4
# Try: batch_size=2, gradient_accumulation=2 (if VRAM allows)

training_args = TrainingArguments(
    ...
    per_device_train_batch_size=2,      # Increase if possible
    gradient_accumulation_steps=2,      # Maintain effective batch=4
    ...
)

# Or try larger batch with gradient accumulation
per_device_train_batch_size=1,
gradient_accumulation_steps=8,  # Effective batch=8
```

**Expected Impact:** More stable gradients, 5-10% improvement  
**GPU Requirement:** Check with `nvidia-smi`

---

### Phase 3: Advanced Optimization (2-5 days)

#### 3.1 **Different Base Models to Test**

| Model | Size | Speed | Accuracy Potential |
|-------|------|-------|-----------|
| Qwen2.5-VL-3B (current) | 3B | ⭐⭐⭐ | ⭐⭐⭐ |
| Qwen2.5-VL-7B | 7B | ⭐⭐ | ⭐⭐⭐⭐ |
| LLaVA-1.6-7B | 7B | ⭐⭐ | ⭐⭐⭐⭐ |
| Phi-3-Vision | 4.2B | ⭐⭐⭐ | ⭐⭐⭐ |

```python
# Try with 7B model for better accuracy
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",  # Upgrade from 3B
    device_map="cuda",
    torch_dtype=torch.float16,
)
```

**Trade-off:** Slower inference but better accuracy

#### 3.2 **Fine-tune Prompt Engineering**
```python
# Current prompt
prompt = "Extract all text from this image."

# Better prompts to try
prompts = [
    "Extract all Odia text from this document image.",
    "Read and transcribe the Odia script text in this image.",
    "OCR: Extract text from image in Odia language.",
    "Transcribe the following document image text accurately.",
    "Extract all visible text exactly as it appears, preserving formatting."
]

# Test which prompt gives best results
for prompt in prompts:
    results = evaluate_with_prompt(prompt)
    print(f"Prompt: {prompt} -> CER: {results['cer']}")
```

#### 3.3 **LoRA Rank & Alpha Tuning**
```python
# Current: r=32, alpha=64
# Try different combinations:

configs = [
    {"r": 16, "alpha": 32},  # Smaller LoRA
    {"r": 32, "alpha": 64},  # Current
    {"r": 64, "alpha": 128}, # Larger LoRA
]

# Larger LoRA = more learning capacity but slower
# Smaller LoRA = faster but less capacity
```

#### 3.4 **Multi-Step Fine-tuning**
```python
# Step 1: Train on general OCR (100 steps)
# Step 2: Fine-tune on Odia-specific data (200 steps)
# Step 3: Fine-tune on complex documents (100 steps)

# This staged approach prevents overfitting
```

---

### Phase 4: Production Optimization (1-2 weeks)

#### 4.1 **Model Quantization** (Speed ↑ 2-3x)
```python
# Reduction 7.5GB → 3.8GB with minimal accuracy loss
# Speed: 433ms → 150ms

from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "shantipriya/qwen2.5-odia-ocr",
    quantization_config=quantization_config,
    device_map="auto"
)
```

**Trade-off:** Slightly lower accuracy (-2-5%) but 3x faster, 50% less memory

#### 4.2 **Distillation** (Size ↓ 60%)
```python
# Compress model through knowledge distillation
# Teacher: Qwen-7B
# Student: Qwen-3B (your model)

# Use teacher's predictions to train student
# Result: Smaller, faster, better accuracy
```

#### 4.3 **Ensemble Methods**
```python
# Combine multiple checkpoints for better predictions

predictions = []
for checkpoint in [50, 100, 150, 200]:
    model = load_checkpoint(f"checkpoint-{checkpoint}")
    pred = model.generate(inputs)
    predictions.append(pred)

# Voting or averaging
final_pred = ensemble_vote(predictions)
```

---

## 🚀 Recommended Quick Implementation (Next 4 hours)

### Step 1: Update training_ocr_qwen.py
```python
training_args = TrainingArguments(
    output_dir="./qwen_ocr_finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    max_steps=500,          # ⬆️ INCREASE from 100
    save_steps=50,
    eval_steps=50,
    evaluation_strategy="steps",  # ⬆️ ADD evaluation
    learning_rate=1e-4,     # ⬇️ LOWER from 2e-4
    lr_scheduler_type="cosine",  # ⬆️ CHANGE from linear
    warmup_steps=50,        # ⬆️ ADD warmup (5-10%)
    fp16=False,
    logging_steps=10,
    logging_dir="./logs",
    save_total_limit=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  # ⬆️ ADD
    report_to="tensorboard",
)
```

### Step 2: Run Improved Training
```bash
ssh root@135.181.8.206 << 'EOF'
cd /root/odia_ocr
source /root/venv/bin/activate
python3 training_ocr_qwen.py 2>&1 | tee training_improved.log
EOF
```

### Step 3: Evaluate Progress
```bash
# After training completes (~5 minutes)
python3 eval_with_examples_v2.py

# Check metrics
cat eval_results/metrics.json
```

---

## 📊 Expected Results After Each Phase

```
Phase 0 (Current):     CER: 100% → Not production-ready
Phase 1 (500 steps):   CER: 30-50% → Noticeable improvement
Phase 2 (1000+ steps): CER: 15-25% → Production-ready
Phase 3 (7B model):    CER: 10-15% → High quality
Phase 4 (Quantized):   CER: 12-18% → Production optimized
```

---

## 🎯 Three-Tier Improvement Plan

### Tier 1: Minimal Effort, Quick Wins ✅
**Time:** 4 hours  
**Cost:** Free (use existing GPU)  
**Expected CER:** 100% → 30-50%

1. Increase max_steps: 100 → 500
2. Add warmup_steps: 50
3. Change lr_scheduler: "cosine"
4. Lower learning_rate: 2e-4 → 1e-4

```bash
# Just update and run:
# Training time: ~5-10 minutes
```

### Tier 2: Solid Improvement, Moderate Effort ✅✅
**Time:** 1-2 days  
**Cost:** Free (use existing data + resources)  
**Expected CER:** 30-50% → 15-25%

1. Collect 500+ training samples
2. Add data augmentation
3. Increase training to 1000+ steps
4. Optimize batch size

```bash
# Multi-day training run
# Training time: ~30-60 minutes
```

### Tier 3: Production Quality, Advanced ✅✅✅
**Time:** 2-5 days  
**Cost:** Compute for larger model  
**Expected CER:** 15-25% → < 10%

1. Try 7B base model
2. Implement knowledge distillation
3. Model quantization
4. Ensemble methods

```bash
# Comprehensive optimization
# Training time: 2-3 hours total
```

---

## 📋 Monitoring & Debugging

### Key Metrics to Track
```python
# During training, monitor:
metrics = {
    "train_loss": "Should decrease steadily",
    "eval_loss": "Should decrease then plateau",
    "learning_rate": "Should decay over time",
    "inference_time": "Should stay consistent",
}

# After training, evaluate:
eval_metrics = {
    "cer": "Character Error Rate (target: < 20%)",
    "wer": "Word Error Rate (target: < 30%)",
    "exact_match": "Exact Match % (target: > 20%)",
    "inference_time": "Should be < 500ms",
}
```

### TensorBoard Monitoring
```bash
# In separate terminal while training
tensorboard --logdir=/root/odia_ocr/logs --port=6006

# Then access: http://localhost:6006
```

---

## 🔧 Troubleshooting Common Issues

### Issue: Loss not decreasing
**Solution:**
- Increase warmup_steps
- Lower learning_rate
- Check data quality
- Enable gradient clipping (already enabled)

### Issue: Model overfitting (eval loss > train loss)
**Solution:**
- Add dropout in data augmentation
- Reduce LoRA rank
- Add early stopping

### Issue: Training too slow
**Solution:**
- Increase batch_size (if VRAM allows)
- Use mixed precision (fp16)
- Reduce max_steps

### Issue: Out of VRAM
**Solution:**
- Reduce batch_size: 1 → 1 (already minimal)
- Reduce gradient_accumulation: 4 → 2
- Use CPU offload or 8-bit quantization

---

## 🎓 Next Steps Priority

1. **TODAY:** Update hyperparameters (30 min)
2. **TODAY:** Run 500-step training (10 min training)
3. **TODAY:** Evaluate results (5 min)
4. **TOMORROW:** Analyze results & collect more data
5. **THIS WEEK:** Implement Phase 2 improvements
6. **NEXT WEEK:** Test Phase 3 optimizations

---

## 📚 Resources & References

- LoRA Paper: https://arxiv.org/abs/2106.09714
- Qwen2.5-VL Model Card: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
- HuggingFace Training Guide: https://huggingface.co/docs/transformers/training
- PEFT Documentation: https://huggingface.co/docs/peft

---

**Last Updated:** February 22, 2026  
**Model Version:** checkpoint-50 with 100 training steps  
**Next Target:** < 20% CER in 500 training steps
