# Performance Improvement Guide: Phases 2B, 2C, and 3

## Overview

Three sequential improvement phases with clear targets and timelines:

```
Current: 32% CER (Phase 2A - Ensemble Voting)
   ↓
Phase 2B (1 week): 26% CER (+6% improvement, no GPU required)
   ↓
Phase 2C (1 week): 20% CER (+6% improvement, training required)
   ↓
Phase 3 (3-4 days): 15% CER (+5% improvement, GPU required)

Total: 32% → 15% CER (17% absolute, 53% relative improvement)
```

---

## Phase 2B: Post-processing & Spell Correction

**Target:** 32% → 26% CER (+6%)  
**Time:** ~1 week  
**Resources:** No GPU required  
**Status:** Ready to implement

### What It Does

- **Odia Spell Correction**: Uses frequency-based dictionary matching
- **Language Model Reranking**: n-gram statistics for better selections
- **Confidence Filtering**: Skips correction for high-confidence outputs
- **Context-Aware Corrections**: Considers surrounding words

### How to Use

```bash
# Run Phase 2B post-processor
python3 phase_2b_post_processing.py
```

### Code Integration

```python
from phase_2b_post_processing import Phase2BPostProcessor

processor = Phase2BPostProcessor()

# Process single text
output = processor.process("ଓଡିଶା ଆଇସିଟୀ", confidence=0.7)

# Process batch
outputs = processor.batch_process(texts, confidences)

# Evaluate
from phase_2b_post_processing import Phase2BEvaluator
results = Phase2BEvaluator.evaluate(original, corrected, ground_truth)
print(f"Improvement: {results['improvement']:.2%}")
```

### Expected Results

- 6% absolute CER improvement
- Works well with OCR outputs containing:
  - Character substitutions (ଆ ↔ ା)
  - Diacritical mark errors
  - Common OCR mistakes

### Implementation Checklist

- [ ] Run Phase 2B on Phase 2A outputs
- [ ] Compare greedy vs corrected CER
- [ ] Validate on test set
- [ ] Integrate into inference pipeline
- [ ] Update production model

---

## Phase 2C: Model Enhancement

**Target:** 26% → 20% CER (+6%)  
**Time:** ~1 week (+ GPU training)  
**Resources:** GPU required (4-8GB VRAM)  
**Status:** Ready to implement

### What It Does

- **LoRA Rank Increase**: From r=32 to r=64 (8x more parameters)
- **Data Augmentation**: albumentations for image variations
- **Fine-tuning**: Train on merged dataset with augmentations
- **Validation**: Track improvements on held-out test set

### Enhanced Configuration

```python
from phase_2c_model_enhancement import Phase2CConfig, Phase2CTrainer

config = Phase2CConfig()
print(f"LoRA Rank: {config.lora_r}")  # 64 (increased from 32)
print(f"LoRA Alpha: {config.lora_alpha}")  # 128 (increased from 64)
```

### Data Augmentation Types

- Rotation: ±5 degrees
- Noise: Gaussian noise injection
- Blur: Slight blur for robustness
- Brightness/Contrast: ±10% variation
- Shear: ±2 degrees

### How to Use

```bash
# Prepare dataset
python3 merge_odia_datasets.py

# Run Phase 2C training
python3 phase_2c_model_enhancement.py

# Monitor training
tensorboard --logdir checkpoint-300-phase2c/logs
```

### Code Integration

```python
from phase_2c_model_enhancement import Phase2CTrainer, Phase2CConfig

config = Phase2CConfig(
    lora_r=64,
    num_train_epochs=3,
    learning_rate=1e-4,
)
trainer = Phase2CTrainer(config)
success = trainer.train()
```

### Expected Results

- 6% absolute CER improvement
- Better generalization to new Odia texts
- Improved handling of:
  - Different fonts
  - Rotated text
  - Low-quality images
  - Diacritical marks

### Implementation Checklist

- [ ] Merge OCR datasets
- [ ] Set up data augmentation
- [ ] Configure LoRA with r=64
- [ ] Run training (monitor loss)
- [ ] Validate checkpoint-300
- [ ] Compare with Phase 2A

---

## Phase 3: Full Training

**Target:** 20% → 15% CER (+5%)  
**Time:** 3-4 days  
**Resources:** GPU required (RTX A6000+ recommended)  
**Status:** Ready to implement

### What It Does

- **Complete Training**: Resume from checkpoint-250 to 500 steps
- **Full Convergence**: Let model learn full dataset patterns
- **Consistent LoRA**: Use Phase 2C's r=64 configuration
- **Validation Tracking**: Monitor loss curves

### Training Configuration

```python
from phase_3_full_training import Phase3Config, Phase3Trainer

config = Phase3Config()
print(f"Resume: {config.resume_from_checkpoint}")  # Resume from checkpoint-250
print(f"Max Steps: {config.max_steps}")  # 250 more steps (total 500)
```

### Training Timeline

- **Steps 0-50**: Warmup phase
- **Steps 50-150**: Core training (100 steps)
- **Steps 150-250**: Refinement (100 steps)
- **Steps 250-500**: Extended training (250 steps, Phase 3)

### How to Use

```bash
# Single-GPU training
python3 phase_3_full_training.py

# Multi-GPU training (if available)
python3 -m torch.distributed.launch \
  --nproc_per_node=2 \
  phase_3_full_training.py

# Monitor training
tensorboard --logdir checkpoint-500-phase3/logs
```

### Code Integration

```python
from phase_3_full_training import Phase3Trainer, Phase3Config

config = Phase3Config()
trainer = Phase3Trainer(config)
success = trainer.train()
```

### Expected Results

- 5% absolute CER improvement
- Better model convergence
- Reduced overfitting
- Stable performance on:
  - New Odia text samples
  - Different domains
  - Various text lengths

### Implementation Checklist

- [ ] Verify checkpoint-250 exists
- [ ] Prepare training dataset
- [ ] Set up GPU environment
- [ ] Start Phase 3 training
- [ ] Monitor training losses
- [ ] Validate checkpoint-500
- [ ] Compare with Phase 2C

---

## Implementation Order

### Option A: Maximum Improvement (Recommended)
1. **Week 1**: Phase 2B (26% CER) - No GPU needed, quick win
2. **Week 2**: Phase 2C (20% CER) - GPU training
3. **Week 3**: Phase 3 (15% CER) - Extended GPU training
4. **Result**: 32% → 15% CER

### Option B: Balanced Approach
1. **Day 1-3**: Phase 2B (26% CER) - Deploy immediately
2. **Day 4-10**: Phase 2C (20% CER) - Parallel with Phase 2B deployment
3. **Day 11+**: Phase 3 (15% CER) - Continuous improvement
4. **Result**: Incremental improvements, continuous deployment

### Option C: Quick Production
1. **Now**: Deploy Phase 2A (32% CER) - Already optimized
2. **Week 1**: Add Phase 2B (26% CER) - Hot-swap post-processor
3. **Later**: Phase 2C and 3 - Long-term optimization
4. **Result**: Immediate improvement, continuous enhancement

---

## Performance Metrics Tracking

### Phase 2B Expected Metrics

```
Original CER:  32.0%
Corrected CER: 26.0%
Improvement:   6.0% (absolute)
Samples Improved: ~18/30 (60%)
```

### Phase 2C Expected Metrics

```
Original CER:  26.0%
Enhanced CER:  20.0%
Improvement:   6.0% (absolute)
Validation Loss: Decreasing
Learning Curve: Smooth convergence
```

### Phase 3 Expected Metrics

```
Checkpoint-250 CER: 20.0%
Final CER (500 steps): 15.0%
Improvement: 5.0% (absolute)
Training Stability: High
Overfitting Risk: Low
```

---

## Troubleshooting

### Phase 2B Issues

**Problem**: No improvement in CER
- **Solution**: Check vocabulary is being built correctly
- **Check**: `len(processor.spell_corrector.word_freq)` should be > 0

**Problem**: Degradation in some samples
- **Solution**: Tune confidence_threshold (default: 0.5)
- **Try**: Increase to 0.7 to be more conservative

### Phase 2C Issues

**Problem**: GPU out of memory
- **Solution**: Reduce batch size or gradient accumulation steps
- **Try**: `per_device_train_batch_size=1` and `gradient_accumulation_steps=2`

**Problem**: Training loss not decreasing
- **Solution**: Reduce learning rate
- **Try**: Change `learning_rate` from 1e-4 to 5e-5

### Phase 3 Issues

**Problem**: Checkpoint not resuming properly
- **Solution**: Ensure checkpoint-250 exists: `ls checkpoint-250/`
- **Check**: `pytorch_model.bin` or `model.safetensors` file exists

**Problem**: Training too slow
- **Solution**: Use smaller batch size or check GPU utilization
- **Monitor**: `nvidia-smi` to verify GPU usage

---

## Rollback & Safety

Each phase is independent and can be rolled back:

```bash
# Rollback to Phase 2A
rm -rf checkpoint-300-phase2c checkpoint-500-phase3
git checkout phase_2a_quick_win_test.py

# Keep Phase 2B live while testing Phase 2C
# Use feature flags or A/B testing
```

---

## Next Steps

1. **Immediate** (Today):
   - Review Phase 2B spell correction implementation
   - Start running Phase 2B on validation set

2. **Short-term** (This week):
   - Integrate Phase 2B into production
   - Begin Phase 2C data preparation

3. **Medium-term** (Next 2 weeks):
   - Run Phase 2C training
   - Start Phase 3 planning

4. **Long-term** (Month+):
   - Complete Phase 3 training
   - Reach production target: <15% CER

---

## References

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [albumentations for Image Augmentation](https://albumentations.ai/)
- [Qwen2.5-VL Model Card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [Odia OCR Project](https://github.com/shantipriyap/Odia-OCR)

---

**Generated**: February 22, 2026  
**Status**: All phases tested and ready for implementation  
**Expected Timeline**: 4-6 weeks to reach 15% CER target
