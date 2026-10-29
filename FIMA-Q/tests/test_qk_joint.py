"""CPU numerical checks using the real FIMA-Q quantized linear/matmul classes."""

import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quant_layers.linear import AsymmetricallyBatchingQuantLinear
from quant_layers.matmul import AsymmetricallyBatchingQuantMatMul
from quantizers.adaround import AdaRoundQuantizer
from utils.qk_joint import (QKParameters, make_work_block, calibrate_qk_block,
                            attention_kl, teacher_attention, dplr_terms,
                            accept_qk_candidate, post_round_teacher_kl, reconstruction_objective)


def linear(inputs, outputs, branches=1):
    layer = AsymmetricallyBatchingQuantLinear(inputs, outputs, n_V=branches, w_bit=3, a_bit=3)
    with torch.no_grad():
        weight = layer.weight.view(branches, outputs // branches, inputs)
        scale = (weight.amax(-1, keepdim=True) - weight.amin(-1, keepdim=True)).clamp_min(1e-8) / 7
        layer.w_quantizer.scale.copy_(scale)
        layer.w_quantizer.zero_point.copy_(-weight.amin(-1, keepdim=True) / scale)
        layer.a_quantizer.scale.fill_(0.4)
        layer.a_quantizer.zero_point.fill_(3)
    layer.calibrated = True
    layer.w_quantizer.inited = layer.a_quantizer.inited = True
    return layer


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads, self.scale = 2, 0.5
        self.q_norm = self.k_norm = nn.Identity()
        self.qkv = linear(8, 24, 3)
        self.proj = linear(8, 8)
        self.matmul1 = AsymmetricallyBatchingQuantMatMul(3, 3, num_heads=2, token_channel_wise=True)
        self.matmul2 = AsymmetricallyBatchingQuantMatMul(3, 3, num_heads=2)
        for matmul in (self.matmul1, self.matmul2):
            for quantizer in (matmul.A_quantizer, matmul.B_quantizer):
                quantizer.scale.data.fill_(0.3)
                quantizer.zero_point.data.fill_(3)
                quantizer.inited = True
            matmul.calibrated = True
        self.matmul1.A_quantizer.scale = nn.Parameter(torch.full((1, 2, 5, 1), 0.3))
        self.matmul1.B_quantizer.scale = nn.Parameter(torch.full((1, 2, 1, 5), 0.3))

    def forward(self, x):
        batch, tokens, _ = x.shape
        q, k, v = self.qkv(x).reshape(batch, tokens, 3, 2, 4).permute(2, 0, 3, 1, 4)
        probability = (self.matmul1(q, k.transpose(-1, -2)) * self.scale).softmax(-1)
        value = self.matmul2(probability, v).transpose(1, 2).reshape(batch, tokens, 8)
        return self.proj(value)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(8), nn.LayerNorm(8)
        self.attn = Attention()
        self.mlp = nn.Sequential(linear(8, 16), nn.GELU(), linear(16, 8))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block()])
        self.head = nn.Linear(8, 4)

    def forward(self, x):
        return self.head(self.blocks[0](x).mean(1))


def mode(block, value):
    for module in block.modules():
        if hasattr(module, "mode"):
            module.mode = value


class Reconstructor:
    def __init__(self, model, x):
        self.model = model
        self.blocks = {"blocks.0": model.blocks[0]}
        self.quanted_blocks = []
        self.k, self.temperature = 2, 20
        self.calib_loader = DataLoader(TensorDataset(x, torch.zeros(len(x))), batch_size=4)
        mode(model, "raw")
        with torch.no_grad():
            self.raw_pred_softmaxs = [(model(x) / self.temperature).softmax(-1)]
            model.blocks[0].raw_input = x.clone()
            model.blocks[0].raw_out = model.blocks[0](x).detach()

    def set_block_mode(self, block, value):
        mode(block, value)


class QKJointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = Model().eval()
        self.block = self.model.blocks[0]
        self.x = torch.randn(12, 5, 8)

    def test_nontrivial_transform_preserves_fp_and_folded_quantization(self):
        for variant in ("A", "B", "C"):
            with self.subTest(variant=variant):
                block = copy.deepcopy(self.block)
                work, fit = make_work_block(block, variant)
                with torch.no_grad():
                    fit.log_q_scale.fill_(0.1)
                    fit.log_k_scale.fill_(-0.1)
                    if variant != "A":
                        fit.log_diag.copy_(torch.linspace(-0.2, 0.2, 8).view(2, 4))
                    if variant == "C":
                        fit.skew.normal_(0, 0.03)
                    mode(block, "raw")
                    before = block(self.x)
                    work_quant = work(self.x)
                    fit.fold_into(block.attn)
                    self.assertTrue(torch.allclose(block(self.x), before, atol=2e-6, rtol=2e-6))
                    mode(block, "quant_forward")
                    self.assertTrue(torch.allclose(block(self.x), work_quant, atol=2e-6, rtol=2e-6))
                self.assertFalse(any("_qk_fit" in key for key in block.state_dict()))

    def test_rotation_is_orthogonal_and_receives_task_gradient(self):
        work, fit = make_work_block(self.block, "C")
        target = self.block(self.x).detach()
        gradient = torch.rand(2, target[0].numel())
        sum(dplr_terms(work(self.x), target, gradient, torch.eye(2))).backward()
        self.assertIsNotNone(fit.skew.grad)
        self.assertGreater(float(fit.skew.grad.abs().sum()), 0)
        with torch.no_grad():
            fit.skew.normal_(0, 0.1)
            rotation = fit.rotation()
            self.assertTrue(torch.allclose(rotation @ rotation.transpose(-1, -2), fit.eye.expand(2, -1, -1), atol=1e-6))

    def test_low_rank_rotation_is_foldable_and_receives_task_gradient(self):
        block = copy.deepcopy(self.block)
        work, fit = make_work_block(block, "C", rotation_rank=1)
        full = QKParameters(block.attn, "C")
        self.assertLess(sum(p.numel() for p in fit.parameters() if p.requires_grad),
                        sum(p.numel() for p in full.parameters() if p.requires_grad))
        target = block(self.x).detach()
        gradient = torch.rand(2, target[0].numel())
        sum(dplr_terms(work(self.x), target, gradient, torch.eye(2))).backward()
        self.assertGreater(float(fit.rot_b.grad.abs().sum()), 0)
        with torch.no_grad():
            fit.rot_b.normal_(0, 0.1)
            rotation = fit.rotation()
            self.assertTrue(torch.allclose(rotation @ rotation.transpose(-1, -2),
                                           fit.eye.expand(2, -1, -1), atol=1e-6))
            mode(block, "raw")
            before = block(self.x)
            work_quant = work(self.x)
            fit.fold_into(block.attn)
            self.assertTrue(torch.allclose(block(self.x), before, atol=2e-6, rtol=2e-6))
            mode(block, "quant_forward")
            self.assertTrue(torch.allclose(block(self.x), work_quant, atol=2e-6, rtol=2e-6))

    def test_relative_kl_gate_rejects_tiny_improvement(self):
        self.assertEqual(accept_qk_candidate(0.526376, 0.525862, 0.05)[0], False)
        self.assertEqual(accept_qk_candidate(0.526376, 0.490, 0.05)[0], True)

    def test_post_round_probe_preserves_modes_and_uses_holdout(self):
        recon = Reconstructor(self.model, self.x)
        before = [(module, module.mode) for module in self.model.modules() if hasattr(module, "mode")]
        measured = post_round_teacher_kl(recon, self.block, 4, 2, torch.device("cpu"))
        self.assertGreaterEqual(measured, 0)
        self.assertTrue(all(module.mode == mode_before for module, mode_before in before))

    def test_attention_target_and_row_shift_invariance(self):
        target = teacher_attention(self.block, self.x)
        scores = target.log()
        shifts = torch.randn(*scores.shape[:-1], 1)
        self.assertAlmostEqual(float(attention_kl(scores, target)), 0, places=6)
        self.assertAlmostEqual(float(attention_kl(scores + shifts, target)), 0, places=6)

    def test_c_fold_preserves_the_quantization_grid_when_entering_adaround(self):
        with torch.no_grad():
            self.block.attn.qkv.w_quantizer.zero_point.round_()
        fit = QKParameters(self.block.attn, "C")
        with torch.no_grad():
            fit.skew.normal_(0, 0.08)
            fit.log_diag.uniform_(-0.2, 0.2)
            fit.fold_into(self.block.attn)
            layer = self.block.attn.qkv
            weight = layer.weight.view(3, 8, 8)
            before = layer.w_quantizer(weight)
            adaround = AdaRoundQuantizer(layer.w_quantizer, weight)
            adaround.soft_targets = False
            after = adaround(weight)
        self.assertTrue(torch.allclose(before, after, atol=1e-7, rtol=1e-7),
                        "Uniform -> AdaRound changed the transformed Q/K quantization grid")

    def test_prepare_runs_all_variants_and_preserves_rng_and_model_modes(self):
        for variant in ("A", "B", "C"):
            with self.subTest(variant=variant):
                model = copy.deepcopy(self.model)
                recon = Reconstructor(model, self.x)
                block = model.blocks[0]
                rng = torch.random.get_rng_state().clone()
                result = calibrate_qk_block(recon, "blocks.0", block, torch.device("cpu"),
                    dict(variant=variant, steps=3, holdout=4, batch_size=4,
                         lr=0.003, eval_every=1, seed=3407))
                self.assertEqual(result["train_samples"], 8)
                self.assertEqual(result["holdout_samples"], 4)
                self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
                self.assertTrue(all(module.mode == "raw" for module in block.modules() if hasattr(module, "mode")))
                self.assertTrue(torch.isfinite(block.attn.qkv.weight).all())

    def test_low_rank_calibration_reports_protocol_and_preserves_rng(self):
        recon = Reconstructor(self.model, self.x)
        rng = torch.random.get_rng_state().clone()
        result = calibrate_qk_block(recon, "blocks.0", self.block, torch.device("cpu"),
                                    dict(variant="C", protocol="v3_lowrank_gate",
                                         rotation_rank=1, min_relative_kl=0.05,
                                         steps=3, holdout=4, batch_size=4,
                                         lr=0.003, eval_every=1, seed=3407))
        self.assertEqual(result["protocol"], "v3_lowrank_gate")
        self.assertEqual(result["rotation_rank"], 1)
        self.assertEqual(result["min_relative_kl"], 0.05)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_loss_is_independent_of_rotation_variant(self):
        self.assertEqual(reconstruction_objective("B"), "mse")
        self.assertEqual(reconstruction_objective("C"), "dplr")
        for variant, loss in (("A", "dplr"), ("B", "dplr"), ("C", "mse")):
            with self.subTest(variant=variant, loss=loss):
                model = copy.deepcopy(self.model)
                recon = Reconstructor(model, self.x)
                result = calibrate_qk_block(recon, "blocks.0", model.blocks[0], torch.device("cpu"),
                    dict(variant=variant, reconstruction_loss=loss, steps=2,
                         holdout=4, batch_size=4, lr=0.003, eval_every=1, seed=3407))
                self.assertEqual(result["reconstruction_loss"], loss)
                self.assertEqual(result["fisher_rank"], 2 if loss == "dplr" else 0)


if __name__ == "__main__":
    unittest.main()
