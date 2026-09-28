# Fisher Probe Pilot Analysis

## Scope

This run is a diagnostic on six ViT-S modules, using 16 images to construct an
error PCA basis and four independent images for evaluation.  All conclusions
are preliminary because the evaluation set contains only four images.

## Main finding

The forward-only central-difference estimator is promising.  The fixed
cross-image low-rank error basis is currently the limiting assumption.

| Module | Rank-8 basis energy on basis images | Rank-8 coverage on evaluation images | Projected/full KL ratio | Best epsilon | Fisher matrix error | Speedup vs. JVP | Incremental memory, FD/JVP |
|---|---:|---:|---:|---:|---:|---:|---:|
| blocks.0.attn | 77.59% | 36.89% | 145.46% | 1e-4 | 0.339% | 3.89x | 4.55/87.22 MiB |
| blocks.0.mlp | 63.47% | 17.87% | 43.45% | 1e-4 | 0.299% | 3.14x | 4.55/84.61 MiB |
| blocks.5.attn | 59.76% | 4.50% | 5.05% | 1e-2 | 0.0366% | 3.31x | 4.55/50.71 MiB |
| blocks.5.mlp | 65.48% | 12.50% | 15.52% | 1e-3 | 0.0657% | 3.35x | 4.55/48.11 MiB |
| blocks.11.attn | 73.32% | 11.61% | 12.47% | 1e-3 | 0.0065% | 3.65x | 4.09/6.71 MiB |
| blocks.11.mlp | 56.72% | 2.06% | 4.99% | 1e-3 | 0.0040% | 3.40x | 2.22/3.25 MiB |

The KL ratio is not an energy coverage metric and can exceed 100% because the
projected perturbation is a different perturbation from the complete
quantization error.  It is included to show how far the projected diagnostic
can be from the true full-error objective.

## Forward-only estimator

A single epsilon gives a 3.1x--3.9x speedup in most rank-8 cases.  It uses 16
suffix forward evaluations, whereas the reference uses eight JVP calls.  The
memory advantage is strongest in early and middle blocks.

Epsilon is depth dependent.  Early blocks prefer 1e-4, middle blocks prefer
1e-3 or 1e-2, and late blocks prefer about 1e-3.  A fixed 1e-3 remains a useful
default: its worst matrix error is 2.72% at block 0 attention, while all other
modules are below 0.7%.  Running all three epsilon values removes much of the
speed advantage, so a final method should select the scale from one or two
cheap calibration examples or use a two-scale consistency rule.

The held-out directional KL tests also separate estimator error from Taylor
approximation error.  Middle and late blocks have very small errors with the
full projected Fisher.  Block 0 attention is an exception: at rank 8 the
autograd reference itself has about 24.2% relative RMSE.  This error therefore
cannot be fixed by improving finite differences alone; the perturbation scale
or quadratic model must also be adjusted for early attention.

## Low-rank and Attention/MLP hypotheses

The error snapshots are not strongly low-rank.  Reaching 95% basis energy needs
rank 14 or 15 out of 16 for every module.  Inside the rank-8 error subspace, the
projected Fisher also needs about 6.5--7.5 dimensions for 95% energy.

The rank-8 off-diagonal ratios are high for every module (about 0.50--0.58).
MLP is often at least as non-diagonal as attention.  Therefore this pilot does
not support a simple rule such as "full covariance for attention, diagonal for
MLP."  The off-diagonal ratio is measured in PCA coordinates, so it should not
be interpreted as an original channel-coordinate statistic.

For the observed quantization-error direction, the relative RMSE of the full
rank-8 projected Fisher is 2.6%--10.6% in four middle/late cases, 0.6% for the
last MLP, and about 33% in both block-0 modules.  Diagonal and half-rank
approximations vary substantially across modules and are usually worse.  The
variation supports an adaptive structure decision, but the current four-image
sample is too small to define a reliable Attention/MLP rule.

## Decision

Continue the forward-only Fisher line, but do not yet integrate the fixed
rank-8 PCA basis into FIMA-Q.  First replace or strengthen the direction set.
A useful next design is a per-batch structured sketch: include the current
quantization-error direction and a small number of head-wise directions for
attention or neuron/channel-group directions for MLP.  Estimate their Fisher
responses with central differences and select epsilon by two-scale
consistency.

The next validation should use at least 32 independent evaluation images and
test basis sizes 16/64/128 with ranks 8/16/32.  A basis is worth integrating
only if evaluation coverage rises materially and the projected objective tracks
full quantization KL.  If coverage stays low, use online structured directions
instead of increasing PCA rank further.

## Interpretation limits

The reported "full" Fisher in prediction_errors.csv is the complete matrix
inside the selected 4-D or 8-D subspace, not the full ambient Fisher matrix.
Timing compares this diagnostic finite-difference primitive with its JVP
reference; it does not yet measure end-to-end FIMA-Q calibration time or final
ImageNet accuracy.
