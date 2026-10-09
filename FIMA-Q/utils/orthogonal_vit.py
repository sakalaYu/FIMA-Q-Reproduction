"""Exact head-wise orthogonal reparameterization for timm Vision Transformers."""

import math

import torch
from torch import nn


def _normalized_hadamard(size: int, *, device, dtype) -> torch.Tensor:
    if size < 1 or size & (size - 1):
        raise ValueError(f"Hadamard head dimension must be a power of two, got {size}")
    matrix = torch.ones((1, 1), device=device, dtype=dtype)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            (torch.cat((matrix, matrix), dim=1),
             torch.cat((matrix, -matrix), dim=1)),
            dim=0,
        )
    return matrix / math.sqrt(size)


def apply_headwise_hadamard(model: nn.Module, verify: bool = True) -> dict:
    """Fold a fixed normalized Hadamard basis into every ViT attention head.

    Q and K are rotated by the same orthogonal matrix, preserving their dot
    products. V uses the same rotation and the inverse is folded into the
    matching input columns of the output projection. No runtime layers or
    trainable parameters are added.
    """
    blocks = getattr(model, "blocks", None)
    if blocks is None or not len(blocks):
        raise TypeError("Expected a timm ViT-like model with non-empty model.blocks")

    probe = None
    reference = None
    was_training = model.training
    if verify:
        # Deterministic input avoids consuming the caller's RNG state.
        parameter = next(model.parameters())
        probe = torch.linspace(-1.0, 1.0, 3 * 224 * 224,
                               device=parameter.device, dtype=parameter.dtype)
        probe = probe.reshape(1, 3, 224, 224)
        model.eval()
        with torch.inference_mode():
            reference = model(probe).detach().float()

    total_heads = 0
    head_dim_seen = set()
    with torch.no_grad():
        for block_index, block in enumerate(blocks):
            attn = getattr(block, "attn", None)
            qkv = getattr(attn, "qkv", None)
            proj = getattr(attn, "proj", None)
            if not isinstance(qkv, nn.Linear) or not isinstance(proj, nn.Linear):
                raise TypeError(f"blocks.{block_index}.attn must have linear qkv and proj")
            for norm_name in ("q_norm", "k_norm"):
                norm = getattr(attn, norm_name, None)
                if norm is not None and not isinstance(norm, nn.Identity):
                    raise ValueError(
                        f"blocks.{block_index}.attn.{norm_name} is non-identity; "
                        "the fixed rotation is not valid across this normalization"
                    )

            heads = int(getattr(attn, "num_heads", 0))
            if heads < 1 or qkv.out_features % (3 * heads):
                raise ValueError(f"Invalid QKV dimensions at blocks.{block_index}.attn")
            head_dim = qkv.out_features // (3 * heads)
            if qkv.out_features != 3 * qkv.in_features or proj.in_features != qkv.in_features:
                raise ValueError(f"Unexpected fused QKV/projection shape at blocks.{block_index}")

            rotation = _normalized_hadamard(
                head_dim, device=qkv.weight.device, dtype=qkv.weight.dtype
            )
            old_weight = qkv.weight.detach().clone()
            old_bias = qkv.bias.detach().clone() if qkv.bias is not None else None
            for branch in range(3):
                for head in range(heads):
                    start = branch * qkv.in_features + head * head_dim
                    stop = start + head_dim
                    qkv.weight[start:stop].copy_(rotation.T @ old_weight[start:stop])
                    if old_bias is not None:
                        qkv.bias[start:stop].copy_(rotation.T @ old_bias[start:stop])

            old_proj_weight = proj.weight.detach().clone()
            for head in range(heads):
                start = head * head_dim
                stop = start + head_dim
                proj.weight[:, start:stop].copy_(old_proj_weight[:, start:stop] @ rotation)

            total_heads += heads
            head_dim_seen.add(head_dim)

    max_error = None
    if verify:
        with torch.inference_mode():
            transformed = model(probe).detach().float()
        max_error = float((reference - transformed).abs().max().item())
        if not torch.allclose(reference, transformed, atol=2e-5, rtol=2e-5):
            model.train(was_training)
            raise RuntimeError(
                f"Full-precision logits changed after rotation (max abs error {max_error:.3e})"
            )
        model.train(was_training)

    return {
        "blocks": len(blocks),
        "heads": total_heads,
        "head_dims": sorted(head_dim_seen),
        "max_logit_abs_error": max_error,
    }
