# Head-wise orthogonal rotation for W4A4 ViT

## Idea

Use a fixed normalized Sylvester Hadamard matrix `H` (so `H H^T = I`) independently in each attention head before quantization. For row-vector activations, transform `Q' = QH`, `K' = KH`, and `V' = VH`. Then

`Q'K'^T = QHH^TK^T = QK^T`.

The attention probabilities are unchanged. If the original attention output is `Y = AV`, then `Y' = YH`. Fold the inverse basis into the matching columns of the output projection: `W_O'[:, head] = W_O[:, head]H`. Thus `Y'(W_O')^T = YW_O^T`. The full-precision function is preserved; the experiment changes only the coordinate basis seen by low-bit quantizers. It adds no runtime operations or learned parameters.

The implementation requires a power-of-two head dimension and identity/absent Q/K normalization. It verifies full-model logits on a deterministic probe before continuing to calibration.

## Experiment protocol

Run baseline and rotation with the same seed, calibration/reconstruction sizes, config, data root, and device. Each run recalibrates from the original pretrained checkpoint, because quantization scales from an unrotated model cannot be reused after rotating weights. The config is the existing `configs/4bit/best.py` (W4A4, Fisher-DPLR block reconstruction).

Start with the smoke pair:

```bash
bash scripts/run_headwise_hadamard.sh smoke
bash scripts/run_headwise_hadamard.sh rotation-smoke
```

Then run the full matched pair:

```bash
bash scripts/run_headwise_hadamard.sh baseline
bash scripts/run_headwise_hadamard.sh rotation
```

Set `DATASET=/path/to/imagenet_fimaq` if the default `../imagenet_fimaq` is not correct. Runs acquire the same `checkpoints/fisher_probe/gpu.lock` used by the diagnostic scripts. Output logs and checkpoints are created under `checkpoints/quant_result/` and mirrored to `logs/`.

Compare validation Top-1/Top-5 and reconstruction time from each `output.log`. The smoke runs only check integration and function preservation; they are not accuracy evidence. A positive result supports this transform for this checkpoint/configuration, but the transform alone is not a novelty claim: orthogonal reparameterization is an active quantization technique, and newer ViT work has also explored rotation-based quantization.
