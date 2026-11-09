"""Protocol separation and completion accounting for fresh LS controls."""
import argparse
import ast
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_lsvit_controls as runner
from utils.lsvit_protocol import validate_lsvit_control


class ControlTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse((ROOT / 'test_quant.py').read_text(encoding='utf-8'))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_args_parser')
        scope = dict(argparse=argparse)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<parser>', 'exec'), scope)
        self.parser = scope['get_args_parser']()
        self.args = runner.get_parser().parse_args(['--worker', '1'])

    def recipe(self, variant):
        return self.parser.parse_args(runner.command(variant, self.args)[3:])

    def test_actual_cli_recipes_are_disjoint_and_preserve_common_settings(self):
        for variant in ('plain', 'fixed', 'b', 'c'):
            a = self.recipe(variant)
            validate_lsvit_control(a)
            self.assertEqual(a.headwise_hadamard, variant != 'plain')
            self.assertEqual(a.qk_joint_variant, {'plain': 'none', 'fixed': 'none', 'b': 'B', 'c': 'C'}[variant])
            self.assertEqual((a.k, a.recon_iters, a.optim_size, a.optim_batch_size), (1, 20000, 1024, 32))
            self.assertEqual((a.seed, a.method_adaptation), (3407, 'lsvit'))
            self.assertFalse(a.load_optimize_checkpoint)

    def test_accidental_extra_rotation_or_C_is_rejected(self):
        a = self.recipe('plain')
        a.headwise_hadamard = True
        with self.assertRaises(ValueError):
            validate_lsvit_control(a)
        a = self.recipe('b')
        a.qk_joint_variant = 'C'
        with self.assertRaises(ValueError):
            validate_lsvit_control(a)

    def test_legacy_C_recipe_remains_valid(self):
        args = runner.qk.get_parser().parse_args(['--variant', 'C', '--reconstruction-loss', 'mse'])
        cmd = runner.qk.command('C', args) + ['--method-adaptation', 'lsvit']
        a = self.parser.parse_args(cmd[3:])
        validate_lsvit_control(a)

    def test_fp_source_supports_safetensors_and_pretrained_fallback(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(runner, 'ROOT', Path(folder).resolve()):
            a = self.recipe('b')
            self.assertIsNone(a.fp_checkpoint)
            self.assertEqual(a.expected_fp_model_sha256, runner.REFERENCE_FP_SHA)
            weights = runner.ROOT / 'checkpoints/vit_raw/model.safetensors'
            weights.parent.mkdir(parents=True)
            weights.touch()
            self.assertEqual(Path(self.recipe('b').fp_checkpoint), weights)
            self.args.fp_checkpoint = Path('weights/alternate.bin')
            self.assertEqual(Path(self.recipe('b').fp_checkpoint), runner.ROOT / self.args.fp_checkpoint)

    def test_control_rejects_checkpoint_or_auxiliary(self):
        for key, value in (('load_calibrate_checkpoint', 'old.pth'), ('qwt', True),
                           ('probe_kd', 'uniform'), ('attention_relation', 'uniform')):
            a = self.recipe('plain')
            setattr(a, key, value)
            with self.assertRaises(ValueError):
                validate_lsvit_control(a)

    def result(self, variant):
        payload = dict(control=variant)
        result = dict(status='complete', launch=payload, control=variant, fp_model_sha256='a' * 64,
            after=dict(top1=65.0, top5=87.0, loss=1.4),
            calibration_hashes=[['128', 'b' * 64, 'c' * 64], ['1024', 'd' * 64, 'e' * 64]],
            cfg=dict(optim_metric='lsvit', k=1, recon_iters=20000, optim_size=1024, calib_size=128,
                     optim_batch_size=32, calib_batch_size=32, optim_mode='qdrop', drop_prob=0.5,
                     w_bit=3, a_bit=3), qk_blocks=[])
        if variant in ('b', 'c'):
            result['qk_blocks'] = [dict(block=f'blocks.{i}', variant=variant.upper(),
                reconstruction_loss='mse', protocol='v2_integer_zp') for i in range(12)]
        return result, payload

    def test_resume_checks_hashes_blocks_metrics_and_recipe(self):
        for variant in ('plain', 'fixed', 'b', 'c'):
            result, payload = self.result(variant)
            self.assertTrue(runner.valid_result(result, payload))
            for key, value in (('status', 'running'), ('control', 'other'),
                               ('calibration_hashes', []), ('after', dict(top1=float('nan')))):
                bad = copy.deepcopy(result)
                bad[key] = value
                self.assertFalse(runner.valid_result(bad, payload))
        b, payload = self.result('b')
        b['qk_blocks'].pop()
        self.assertFalse(runner.valid_result(b, payload))
        plain, payload = self.result('plain')
        payload['expected_fp_model_sha256'] = runner.REFERENCE_FP_SHA
        self.assertFalse(runner.valid_result(plain, payload))
        plain['fp_model_sha256'] = runner.REFERENCE_FP_SHA
        self.assertTrue(runner.valid_result(plain, payload))
        plain['qk_blocks'] = [dict(block='blocks.0')]
        self.assertFalse(runner.valid_result(plain, payload))

    def test_minimal_LS_controls_reconstruct_real_quantized_blocks(self):
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        from types import SimpleNamespace
        import test_endpoint_numerics as endpoint
        from test_qk_joint import Block
        from utils.orthogonal_vit import apply_headwise_hadamard
        cls = endpoint.EndpointNumerics().reconstructor_class()
        dummy = type('UnusedTimm', (nn.Module,), {})
        cls.__init__.__globals__['timm'] = SimpleNamespace(
            layers=SimpleNamespace(patch_embed=SimpleNamespace(PatchEmbed=dummy)),
            models=SimpleNamespace(vision_transformer=SimpleNamespace(Block=Block),
              swin_transformer=SimpleNamespace(SwinTransformerBlock=dummy, PatchMerging=dummy)))
        class Mini(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([Block()])
                b = self.blocks[0]
                b.ls1 = b.ls2 = b.drop_path1 = b.drop_path2 = nn.Identity()
                self.readout = nn.Linear(8, 4)
            def forward(self, x):
                return self.readout(self.blocks[0](x).mean(1))
        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for variant in ('plain', 'fixed', 'b'):
                torch.manual_seed(14)
                model = Mini().eval()
                if variant != 'plain':
                    apply_headwise_hadamard(model, verify=False)
                for module in model.modules():
                    if hasattr(module, 'w_quantizer'):
                        module.w_quantizer.zero_point.data.round_()
                x = torch.randn(12, 5, 8)
                loader = DataLoader(TensorDataset(x, torch.zeros(12)), batch_size=4)
                qk = (dict(variant='B', reconstruction_loss='mse', protocol='v2_integer_zp',
                     steps=2, batch_size=4, holdout=4, lr=0.003, eval_every=1, seed=3407)
                     if variant == 'b' else None)
                recon = cls(model, 4, loader, metric='lsvit', k=1, iters=2, qk_joint_options=qk)
                recon.reconstruct_model(quant_act=True, mode='qdrop', drop_prob=0.5, keep_gpu=False)
                self.assertTrue(bool(torch.isfinite(model(x)).all()))
                self.assertEqual(len(recon.qk_joint_results), 1 if variant == 'b' else 0)
        finally:
            torch.set_num_threads(original_threads)


if __name__ == '__main__':
    unittest.main()
