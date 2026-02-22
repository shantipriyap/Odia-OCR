# Quick Start: Performance Improvements

## Today: Phase 2B (No GPU)

```bash
# Run the post-processor
python3 phase_2b_post_processing.py

# Expected output:
# ✅ Post-processor initialized
# 📝 Test Examples: 3 Odia sentences processed
# ✅ Results saved to phase_2b_results.json
```

**Expected CER**: 32% → 26% ✅

---

## This Week: Phase 2C (GPU Required)

```bash
# Merge datasets
python3 merge_odia_datasets.py

# Run Phase 2C
python3 phase_2c_model_enhancement.py

# Monitor training
tensorboard --logdir checkpoint-300-phase2c/logs
```

**Expected CER**: 26% → 20% ✅

---

## Next Week: Phase 3 (GPU Required)

```bash
# Single GPU
python3 phase_3_full_training.py

# Multi GPU (recommended)
python3 -m torch.distributed.launch \
  --nproc_per_node=2 \
  phase_3_full_training.py

# Monitor training
tensorboard --logdir checkpoint-500-phase3/logs
```

**Expected CER**: 20% → 15% ✅

---

## Verification Checklist

### Phase 2B
- [ ] Script runs without errors
- [ ] Results saved to phase_2b_results.json
- [ ] CER improves by ~6%
- [ ] Integration tested

### Phase 2C
- [ ] Checkpoint-300 created
- [ ] Training loss decreasing
- [ ] Validation CER improves
- [ ] Model converges

### Phase 3
- [ ] Checkpoint-500 created
- [ ] Training stable
- [ ] Final CER ~15%
- [ ] Model deployment ready

---

## Performance Timeline

```
Day 1:    Phase 2B complete → 26% CER
Day 8:    Phase 2C complete → 20% CER
Day 18:   Phase 3 complete → 15% CER

Total: 3 weeks to 53% relative improvement
```

---

## Resource Requirements

| Phase | GPU | Time | Batch Size |
|-------|-----|------|-----------|
| 2B    | ❌  | 30m  | N/A       |
| 2C    | ✅  | 7d   | 4-8       |
| 3     | ✅  | 3-4d | 8-16      |

---

## Rollback Commands

```bash
# Keep Phase 2B, remove 2C/3
rm -rf checkpoint-300-phase2c checkpoint-500-phase3

# Restart from beginning
git checkout HEAD -- *.py
rm -rf checkpoint-*

# Deploy only Phase 2B
cp phase_2b_post_processing.py production/
```

---

See [PERFORMANCE_IMPROVEMENT_GUIDE.md](PERFORMANCE_IMPROVEMENT_GUIDE.md) for detailed documentation.
