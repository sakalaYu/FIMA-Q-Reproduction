# Fisher-weighted attention relation experiment

## Question

The fixed per-head Hadamard rotation improved W3A3 ViT-S by about 1.14 Top-1
points over the matched unrotated baseline across three seeds. The A/B/C
follow-up did not beat the fixed-rotation result at seed 3407. The next
high-value test asks whether the original DPLR/AdaRound optimization can use
the teacher's attention relations more effectively when it prioritizes
query/head relations by their existing task gradients.

The experiment has three matched arms, all W3A3, ViT-S, seed 3407, rank 5,
20,000 reconstruction steps, and the same fixed rotation:

| Window | Arm | Difference from the control |
|---|---|---|
| 1 | Control | Fixed rotation only |
| 2 | Uniform | Add uniformly weighted attention-relation KL |
| 3 | Fisher | Weight each head/query relation using gradients collected in FIMA-Q's existing Fisher refreshes |

The teacher attention targets are cached during the existing FP block-cache
pass. The student relation KL is added to the same reconstruction step as the
DPLR and AdaRound terms. The Fisher arm reuses gradients from existing FIMA-Q
refreshes; it adds no optimizer steps or model backward passes. The relation
term warms in from 20% to 60% of the reconstruction schedule, and its loss
value is capped at 10% of the current reconstruction loss.

The Fisher weights are defined per attention head and selected query row.
They are derived from the squared gradient of the existing classification KL
with respect to attention scores, averaged over calibration samples, clipped,
smoothed toward uniform weights, and normalized to mean 1. The uniform arm
isolates whether this task weighting helps beyond adding an attention loss.

## Novelty boundary

Attention-score reconstruction already exists in ADFQ-ViT. Do not describe
adding an attention KL as the contribution. The narrower hypothesis to test is
whether FIMA-Q's task Fisher identifies which head/query relations matter, and
whether using those weights inside the same DPLR/AdaRound objective improves
the quantized model. A positive single-seed result is a prototype signal, not
paper-level evidence; it would still need matched seeds and comparison against
the relevant attention-score reconstruction baseline.

## Server launch

Sync these files to the server before starting:

- `test_quant.py`
- `utils/block_recon.py`
- `utils/attention_relation.py`
- `scripts/run_headwise_experiments.py`
- `scripts/run_attention_relation_experiments.py`

In three tmux windows, use the same GPU slot count in each window:

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q

# Window 1: matched fixed-rotation control
python -u scripts/run_attention_relation_experiments.py --worker 1 --workers 3 --gpu 0 --gpu-slots 3 --resume

# Window 2: uniform relation KL
python -u scripts/run_attention_relation_experiments.py --worker 2 --workers 3 --gpu 0 --gpu-slots 3 --resume

# Window 3: Fisher-weighted relation KL
python -u scripts/run_attention_relation_experiments.py --worker 3 --workers 3 --gpu 0 --gpu-slots 3 --resume
```

Once the jobs finish, summarize without launching another GPU job:

```bash
python scripts/run_attention_relation_experiments.py --variant summary
```

To check commands without starting training:

```bash
python scripts/run_attention_relation_experiments.py --worker 1 --workers 3 --dry-run
python scripts/run_attention_relation_experiments.py --worker 2 --workers 3 --dry-run
python scripts/run_attention_relation_experiments.py --worker 3 --workers 3 --dry-run
```

Each full arm should take roughly 2.75–3 hours when run alone, based on the
recent seed-3407 reconstructions. Sharing one GPU across three runs can take
longer and has a higher memory peak. If any arm OOMs, restart it with
`--gpu-slots 1`; the launcher will queue it behind other launcher jobs.

## Artifacts and decision rule

The launcher writes console captures and `summary_seed3407_relation_v1.json`
under `checkpoints/attention_relation/`. Per-run attention diagnostics are
saved as `attention_relation.json` beside the quantized checkpoint.

Compare each arm with both the fresh control and the prior 64.650% fixed-
rotation result. The calibration hashes, recipe, and worker count must match
before reporting a paired delta. Keep the Fisher arm only if it beats both
control and uniform; if only uniform improves, the result supports attention
relation reconstruction but not Fisher weighting. If neither improves, stop
this branch instead of sweeping many coefficients.
