# Signed task-aligned attention relation: one W3A3 trial

## Why this trial

On 2026-10-21, the matched fixed-rotation control scored 64.524% Top-1.
Uniform attention KL scored 64.488% (-0.036 pp), while magnitude-weighted
Fisher attention KL scored 64.056% (-0.468 pp). All 12 blocks and 16,000
relation steps per block executed. The gradient-magnitude weights therefore
did not convert local relation reconstruction into better classification.

The new trial changes the **criterion for using an attention relation**.
During each existing FIMA-Q Fisher refresh, let `g` be the classification KL
gradient with respect to current attention scores, and let `r = p_quant -
p_FP` be the attention-relation KL gradient. The cosine between `g` and `r`
is averaged over calibration samples for each head/query. A positive average
means a small descent step on relation KL also descends classification KL in
score space. Only those head/query relations receive auxiliary loss weight.
The mask is bounded at weight 2 and updated at all five existing refreshes.

This is a local directional proxy, not proof that the final quantizer update
will improve Top-1. The classifier's gradients are reused; there is no new
full-network backward, model parameter or inference operation. The relation
KL is still optimized inside the original DPLR/AdaRound loop, with the same
fixed Hadamard rotation, 20,000 reconstruction steps and 0.1 loss budget.

## Files to sync

Sync these small code files to the existing server checkout:

- `test_quant.py`
- `utils/attention_relation.py`
- `utils/block_recon.py`
- `scripts/run_aligned_relation_experiment.py`
- `scripts/run_headwise_experiments.py`

The existing `utils/orthogonal_vit.py` and `utils/datasets.py` from the
previous fixed-rotation runs remain required.

## Run in one tmux window

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
python -u scripts/run_aligned_relation_experiment.py --gpu 0 --gpu-slots 1 --resume
```

This reuses the completed seed-3407 relation control and the earlier
64.650% fixed-rotation run as references. It does not start another control
or previous A/B/C job. The run name and saved protocol are
`relation_v2_alignment`, so `--resume` cannot skip an earlier v1 experiment.
Based on the three recent full reconstructions, expect approximately 2.5 to
3 hours on an otherwise idle GPU; the actual time depends on GPU sharing.

After completion:

```bash
python scripts/run_aligned_relation_experiment.py --summary
```

Read `checkpoints/attention_relation/summary_seed3407_relation_v2_alignment.json`
for Top-1 and paired deltas. Read `attention_relation.json` beside the new
quantized checkpoint for each block's mask fraction and gradient refreshes.
The launcher requires the calibration sample and tensor hashes to match
before reporting paired accuracy changes.

Keep this line of work only if the signed alignment trial beats the fresh
64.524% control and substantially improves over the 64.056% magnitude-based
Fisher result. A promising single-seed result needs replication before a
claim of stable improvement. If it does not beat control, stop attention
relation weighting and keep fixed rotation as the best verified method.
