"""CPU checks on real quantizers, inference-path repair and QwT serialization."""
import copy
import ast
from contextlib import nullcontext
import json
import logging
import math
from pathlib import Path
import subprocess
import sys
from types import MethodType, SimpleNamespace
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quant_layers.linear import AsymmetricallyBatchingQuantLinear
from quantizers.adaround import AdaRoundQuantizer
from utils.hard_scale_repair import repair_block_scales, path_diagnostics, path_mse
from utils.qwt_compensation import RegressionStats, fit_qwt, attach_saved_qwt
from utils.calibrator import QuantCalibrator
from quant_layers import MinMaxQuantLinear, MinMaxQuantConv2d, MinMaxQuantMatMul


class QuantBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = AsymmetricallyBatchingQuantLinear(4, 4, w_bit=3, a_bit=3)
        layer = self.linear
        with torch.no_grad():
            layer.weight.copy_(torch.eye(4) + torch.randn(4, 4) * 0.1)
            layer.bias.zero_()
            w = layer.weight.view(1, 4, 4)
            s = (w.amax(-1, keepdim=True) - w.amin(-1, keepdim=True)) / 7
            layer.w_quantizer.scale.copy_(s)
            layer.w_quantizer.zero_point.copy_((-w.amin(-1, keepdim=True) / s).round())
            layer.a_quantizer.scale.fill_(0.8)
            layer.a_quantizer.zero_point.fill_(3)
        layer.w_quantizer.inited = layer.a_quantizer.inited = True
        layer.calibrated = True
        layer.w_quantizer = AdaRoundQuantizer(layer.w_quantizer, w)
        layer.w_quantizer.soft_targets = True
        layer.mode = 'raw'

    def forward(self, x):
        return self.linear(x)


class ToyBlock(nn.Module):
    def __init__(self, matrix, bias):
        super().__init__()
        self.register_buffer('matrix', matrix)
        self.register_buffer('offset', bias)

    def forward(self, x):
        return x @ self.matrix + self.offset


class ToyModel(nn.Module):
    def __init__(self, matrices):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.blocks = nn.ModuleList([ToyBlock(m, torch.arange(4).float() * 0.02) for m in matrices])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class EndpointNumerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def block(self):
        torch.manual_seed(17)
        block = QuantBlock().eval()
        x = torch.randn(64, 5, 4) * 0.8
        block.raw_input = x
        block.quanted_input = x + 0.02 * torch.randn_like(x)
        with torch.no_grad():
            block.raw_out = block(x)
        return block

    def test_scale_repair_updates_real_hard_path_without_changing_weights(self):
        block = self.block()
        before = {n: p.detach().clone() for n, p in block.named_parameters() if 'a_quantizer.scale' not in n}
        original_scale = block.linear.a_quantizer.scale.detach().clone()
        original_flags = {n: p.requires_grad for n, p in block.named_parameters()}
        state = torch.random.get_rng_state().clone()
        report = repair_block_scales(block, 'blocks.0', torch.device('cpu'),
            dict(steps=100, lr=0.02, bound=1.2, holdout=16, batch_size=8, eval_every=10, seed=3407))
        self.assertTrue(report['accepted'])
        self.assertLess(report['hard_mse_after'], report['hard_mse_before'])
        self.assertEqual(report['active_steps'], 100)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertEqual(block.linear.mode, 'raw')
        self.assertTrue(block.linear.w_quantizer.soft_targets)
        self.assertFalse(any('parametrizations' in n for n in block.state_dict()))
        for name, value in before.items():
            self.assertTrue(torch.equal(value, dict(block.named_parameters())[name]))
        self.assertEqual(original_flags, {n: p.requires_grad for n, p in block.named_parameters()})
        ratio = block.linear.a_quantizer.scale / original_scale
        self.assertTrue(bool(((ratio >= 1 / 1.2 - 1e-6) & (ratio <= 1.2 + 1e-6)).all()))
        actual = path_mse(block, list(range(48, 64)), torch.device('cpu'), 8)
        self.assertAlmostEqual(actual, report['hard_mse_after'], places=7)

    def test_repair_accepts_mixed_scale_signs_and_preserves_them(self):
        block = self.block()
        quantizer = block.linear.a_quantizer
        original = torch.tensor([0.8, -0.8, 0.8, -0.8, 0.8]).view(1, 5, 1)
        quantizer.scale = nn.Parameter(original.clone())
        report = repair_block_scales(block, 'blocks.0', torch.device('cpu'),
            dict(steps=20, lr=0.02, bound=1.2, holdout=16, batch_size=8, eval_every=5, seed=3407))
        self.assertEqual(report['active_steps'], 20)
        self.assertEqual(report['negative_scale_counts'], {'linear.a_quantizer': 2})
        self.assertTrue(torch.equal(quantizer.scale.sign(), original.sign()))
        self.assertGreaterEqual(report['ratio_min'], 1 / 1.2 - 1e-6)
        self.assertLessEqual(report['ratio_max'], 1.2 + 1e-6)
        self.assertLessEqual(report['hard_mse_after'], report['hard_mse_before'])

    def test_hardening_diagnostic_preserves_rng_and_saved_path(self):
        block = self.block()
        rng = torch.random.get_rng_state().clone()
        result = path_diagnostics(block, list(range(48, 64)), torch.device('cpu'), 8, 1)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(block.linear.w_quantizer.soft_targets)
        self.assertEqual(block.linear.mode, 'raw')
        self.assertGreater(result['hard_full_mse'], 0)
        self.assertNotEqual(result['soft_full_mse'], result['hard_full_mse'])

    def test_regression_recovers_known_affine_residual_with_singular_features(self):
        torch.manual_seed(7)
        x = torch.randn(20, 5, 4)
        x[..., 3] = x[..., 0] + x[..., 1]  # Rank-deficient feature matrix.
        target = x @ torch.randn(4, 4) + torch.randn(4)
        stats = RegressionStats(4)
        stats.update(x[:8], target[:8])
        stats.update(x[8:], target[8:])
        w, b, report = stats.solve()
        self.assertTrue(report['accepted'])
        self.assertLess(float((nn.functional.linear(x, w, b) - target).square().mean()), 1e-10)

    def test_qwt_sequential_fit_and_state_reload_reproduce_outputs(self):
        torch.manual_seed(31)
        x = torch.randn(24, 5, 4)
        matrices = [torch.eye(4) + torch.randn(4, 4) * 0.05 for _ in range(2)]
        teacher = ToyModel(matrices).eval()
        student = ToyModel([m * 0.8 for m in matrices]).eval()
        loader = DataLoader(TensorDataset(x, torch.zeros(len(x))), batch_size=4)
        old_error = float((student(x) - teacher(x)).square().mean())
        report = fit_qwt(student, teacher, loader, 24, 4, branch_dtype='fp32', solver_threads=1)
        self.assertLess(float((student(x) - teacher(x)).square().mean()), old_error * 1e-5)
        self.assertEqual(len(report['blocks']), 2)
        self.assertEqual(report['added_parameters'], 40)
        self.assertFalse(student.blocks[0]._forward_pre_hooks)
        state = copy.deepcopy(student.state_dict())
        restored = ToyModel([m * 0.8 for m in matrices]).eval()
        attach_saved_qwt(restored, state)
        restored.load_state_dict(state)
        self.assertTrue(torch.equal(student(x), restored(x)))

    def test_fp16_branch_is_in_the_measured_forward(self):
        torch.manual_seed(4)
        x = torch.randn(12, 5, 4)
        teacher = ToyModel([torch.eye(4)]).eval()
        student = ToyModel([torch.eye(4) * 0.9]).eval()
        loader = DataLoader(TensorDataset(x, torch.zeros(len(x))), batch_size=4)
        report = fit_qwt(student, teacher, loader, 12, 4, solver_threads=1)
        self.assertEqual(student.blocks[0].qwt_weight.dtype, torch.float16)
        self.assertEqual(student(x).dtype, x.dtype)
        self.assertLess(report['blocks'][0]['deployed_training_mse'], 1e-5)

    def reconstructor_class(self, source=None):
        # Exercise the actual class/method bodies without importing optional
        # timm in this CPU-only environment. __init__ (model discovery) is unused.
        source = source or (Path(__file__).resolve().parents[1] / 'utils/block_recon.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        classes = [node for node in tree.body if isinstance(node, ast.FunctionDef) or
                   isinstance(node, ast.ClassDef) and node.name in ('BlockReconstructor', 'LossFunction', 'LinearTempDecay')]
        namespace = dict(torch=torch, nn=nn, math=math, logging=logging, json=json,
                         nullcontext=nullcontext, QuantCalibrator=QuantCalibrator,
                         AdaRoundQuantizer=AdaRoundQuantizer, MinMaxQuantLinear=MinMaxQuantLinear,
                         MinMaxQuantConv2d=MinMaxQuantConv2d, MinMaxQuantMatMul=MinMaxQuantMatMul,
                         MethodType=MethodType, F=nn.functional)
        exec(compile(ast.Module(body=classes, type_ignores=[]), '<block_recon CPU integration>', 'exec'), namespace)
        return namespace['BlockReconstructor']

    def train_block(self, block, cls, options):
        recon = cls.__new__(cls)
        for key, value in dict(probe_kd_options=None, endpoint_options=options,
                               endpoint_results=[], current_relation=None, metric='mse',
                               k=1, p1=1.0, p2=1.0, dis_mode='q',
                               robust_beta=0.5, robust_temperature=0.5).items():
            setattr(recon, key, value)
        block.raw_grad = None
        recon.reconstruct_single_block('blocks.0', block, torch.device('cpu'),
                                       batch_size=8, iters=2, quant_act=True, mode='qdrop', drop_prob=0.5)
        return recon

    def test_real_reconstruction_method_runs_both_endpoint_modes(self):
        cls = self.reconstructor_class()
        for mode in ('hard_scale', 'soft_extra'):
            with self.subTest(mode=mode):
                block = self.block()
                recon = self.train_block(block, cls, dict(mode=mode, steps=2, lr=0.001,
                    bound=1.2, holdout=16, batch_size=8, eval_every=1, seed=3407))
                self.assertEqual(len(recon.endpoint_results), 1)
                self.assertEqual(recon.endpoint_results[0]['mode'], mode)
                self.assertFalse(block.linear.w_quantizer.soft_targets)
                self.assertEqual(block.linear.a_quantizer.drop_prob, 1.0)
                self.assertFalse(block.linear.a_quantizer.training_mode)
                self.assertEqual(block.linear.mode, 'raw')

    def test_disabled_endpoint_preserves_original_training_trajectory(self):
        root = Path(__file__).resolve().parents[2]
        process = subprocess.run(['git', 'show', 'HEAD:FIMA-Q/utils/block_recon.py'], cwd=root,
                                 text=True, capture_output=True)
        if process.returncode:
            self.skipTest('Original Git baseline unavailable in this checkout')
        source = process.stdout
        old_cls, new_cls = self.reconstructor_class(source), self.reconstructor_class()
        block = self.block()
        other = copy.deepcopy(block)
        torch.manual_seed(99)
        self.train_block(block, old_cls, None)
        torch.manual_seed(99)
        self.train_block(other, new_cls, None)
        for name, value in block.state_dict().items():
            self.assertTrue(torch.equal(value, other.state_dict()[name]), name)

    def test_full_tiny_attention_reconstruction_then_qwt_on_real_quantizers(self):
        # A real tiny attention/MLP quantized block, actual DPLR refresh, C-MSE
        # pre-calibration, hard repair, final hard-weight baking, then QwT.
        # Only timm type discovery is substituted; numerical methods are real.
        from test_qk_joint import Block as TinyAttentionBlock
        class MiniModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([TinyAttentionBlock()])
                block = self.blocks[0]
                block.ls1 = block.ls2 = block.drop_path1 = block.drop_path2 = nn.Identity()
                self.readout = nn.Linear(8, 4)

            def forward(self, x):
                return self.readout(self.blocks[0](x).mean(1))

        cls = self.reconstructor_class()
        globals_ = cls.__init__.__globals__
        dummy = type('UnusedTimmType', (nn.Module,), {})
        globals_['timm'] = SimpleNamespace(layers=SimpleNamespace(patch_embed=SimpleNamespace(PatchEmbed=dummy)),
            models=SimpleNamespace(vision_transformer=SimpleNamespace(Block=TinyAttentionBlock),
                                   swin_transformer=SimpleNamespace(SwinTransformerBlock=dummy, PatchMerging=dummy)))
        torch.manual_seed(61)
        model = MiniModel().eval()
        teacher = copy.deepcopy(model).eval()
        for module in model.modules():
            if hasattr(module, 'w_quantizer'):
                module.w_quantizer.zero_point.data.round_()
        x = torch.randn(12, 5, 8)
        loader = DataLoader(TensorDataset(x, torch.zeros(len(x))), batch_size=4)
        options = dict(mode='hard_scale', steps=2, lr=0.001, bound=1.2,
                       holdout=4, batch_size=4, eval_every=1, seed=3407)
        qk = dict(variant='C', reconstruction_loss='mse', protocol='v2_integer_zp',
                  steps=2, batch_size=4, holdout=4, lr=0.003, eval_every=1, seed=3407)
        recon = cls(model, 4, loader, metric='fisher_dplr', k=1, iters=2,
                    endpoint_options=options, qk_joint_options=qk)
        recon.reconstruct_model(quant_act=True, mode='qdrop', drop_prob=0.5, keep_gpu=False)
        self.assertEqual(len(recon.endpoint_results), 1)
        self.assertEqual(len(recon.qk_joint_results), 1)
        for module in model.modules():
            if hasattr(module, 'w_quantizer'):
                self.assertFalse(hasattr(module.w_quantizer, 'alpha'))
                self.assertEqual(module.w_quantizer.round_mode, 'nearest')
        before = model(x).detach()
        self.assertTrue(bool(torch.isfinite(before).all()))
        report = fit_qwt(model, teacher, loader, 8, 4, solver_threads=1)
        self.assertEqual(len(report['blocks']), 1)
        self.assertTrue(bool(torch.isfinite(model(x)).all()))


if __name__ == '__main__':
    unittest.main()
