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
        for variant in ('plain', 'fixed', 'b', 'c', *runner.MIX_VARIANTS):
            a = self.recipe(variant)
            validate_lsvit_control(a)
            self.assertEqual(a.headwise_hadamard, variant != 'plain')
            self.assertEqual(a.qk_joint_variant, {'b': 'B', 'c': 'C'}.get(variant, 'none'))
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

    def test_mix_completion_requires_all_probes_and_the_correct_applied_weights(self):
        for variant, weight in (('mix-equal', 0.5), ('mix-adaptive', 0.75)):
            result, payload = self.result(variant)
            result['ls_mix_blocks'] = [dict(block=f'blocks.{i}', step=10000, probe_samples=128,
                applied_weight=weight, suggested_weight=0.75) for i in range(12)]
            self.assertTrue(runner.valid_result(result, payload))
            result['ls_mix_blocks'][0]['applied_weight'] = 0.625
            self.assertFalse(runner.valid_result(result, payload))

    def test_paired_launch_uses_one_checkpoint_and_requires_a_first_branch_signature(self):
        self.args.pair_id = 'pair3407'
        prepare = self.recipe('mix-prepare')
        validate_lsvit_control(prepare)
        self.assertTrue(prepare.calibrate)
        self.assertFalse(prepare.optimize)
        for mode in runner.MIX_VARIANTS:
            branch = self.recipe(mode)
            validate_lsvit_control(branch)
            self.assertEqual(branch.ls_pair_dir, prepare.ls_pair_dir)
            self.assertEqual(Path(branch.load_calibrate_checkpoint), Path(prepare.ls_pair_dir) / 'calibrated.pth')

        result, payload = self.result('mix-equal')
        payload['command'] = ['--ls-pair-dir', prepare.ls_pair_dir]
        result['ls_mix_blocks'] = [dict(block=f'blocks.{i}', step=10000, probe_samples=128,
            applied_weight=0.5, suggested_weight=0.75) for i in range(12)]
        self.assertFalse(runner.valid_result(result, payload))
        signature = {name + '_sha256': 'a' * 64 for name in ('snapshot', 'model', 'optimizer', 'ls', 'rng', 'probe')}
        signature.update(step=10000, data_hashes=result['calibration_hashes'][1:])
        result['pair_start'] = result['ls_mix_blocks'][0]['pair_start'] = signature
        self.assertTrue(runner.valid_result(result, payload))

    def test_shared_snapshot_and_barrier_reject_mismatched_state(self):
        import torch
        from utils.lsvit_pair import fingerprint, read_snapshot, save_snapshot, verify_first_branch

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            initial = directory / 'source.pth'
            data_hashes = [['128', 'a' * 64, 'b' * 64]]
            torch.save({'scale': torch.tensor(1.)}, initial)
            save_snapshot(directory, initial, 'c' * 64, data_hashes, 3407)
            shared = directory / 'calibrated.pth'
            self.assertEqual(read_snapshot(directory, shared, 'c' * 64, 3407)['calibration_hashes'], data_hashes)
            with self.assertRaisesRegex(RuntimeError, 'snapshot fingerprint mismatch'):
                read_snapshot(directory, shared, 'c' * 64, 3408)
            first = {'probe_sha256': fingerprint(torch.tensor(1.))}
            with self.assertRaises(TimeoutError):
                verify_first_branch(directory, 'mix-equal', first, timeout=0)
            verify_first_branch(directory, 'mix-adaptive', first, timeout=0)
            verify_first_branch(directory, 'mix-equal', first, timeout=0)

        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(TimeoutError):
                verify_first_branch(folder, 'mix-equal', first, timeout=0)
            other = {'probe_sha256': fingerprint(torch.tensor(2.))}
            with self.assertRaisesRegex(RuntimeError, 'before adaptation: probe_sha256'):
                verify_first_branch(folder, 'mix-adaptive', other, timeout=0)
            with self.assertRaisesRegex(RuntimeError, 'before adaptation: probe_sha256'):
                verify_first_branch(folder, 'mix-equal', first, timeout=0)

    def test_mix_recovers_known_weight_and_keeps_equal_when_components_are_redundant(self):
        import torch
        from utils.lsvit_adapter import fit_ls_mix

        diag = torch.tensor([1., 2., 3., 4., 1., 2., 3., 4.])
        rank_one = torch.tensor([4., 1., 2., 3., 4., 1., 2., 3.])
        self.assertEqual(fit_ls_mix(diag, rank_one, 2 * (0.25 * diag + 0.75 * rank_one))[0], 0.75)
        self.assertEqual(fit_ls_mix(diag, 2 * diag, 3 * diag)[0], 0.5)

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
                self.blocks = nn.ModuleList([Block(), Block()])
                for block in self.blocks:
                    block.ls1 = block.ls2 = block.drop_path1 = block.drop_path2 = nn.Identity()
                self.readout = nn.Linear(8, 4)
            def forward(self, x):
                for block in self.blocks:
                    x = block(x)
                return self.readout(x.mean(1))
        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for variant in ('plain', 'fixed', 'b', *runner.MIX_VARIANTS):
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
                mix = dict(mode=variant, seed=3407) if variant in runner.MIX_VARIANTS else None
                recon = cls(model, 4, loader, metric='lsvit', k=1, iters=2,
                            qk_joint_options=qk, ls_mix_options=mix)
                recon.reconstruct_model(quant_act=True, mode='qdrop', drop_prob=0.5, keep_gpu=False)
                self.assertTrue(bool(torch.isfinite(model(x)).all()))
                self.assertEqual(len(recon.qk_joint_results), 2 if variant == 'b' else 0)
                self.assertEqual(len(recon.ls_mix_results), 2 if mix else 0)
                if mix:
                    for probe in recon.ls_mix_results:
                        self.assertGreater(probe['mean_kl'], 0)
                        self.assertEqual(probe['step'], 1)
                        weight = 0.5 if variant == 'mix-equal' else probe['suggested_weight']
                        self.assertEqual(probe['applied_weight'], weight)
                if variant == 'fixed':
                    baseline = model.state_dict()
                if variant == 'mix-equal':
                    for name, value in model.state_dict().items():
                        self.assertTrue(torch.equal(value, baseline[name]), name)
        finally:
            torch.set_num_threads(original_threads)


if __name__ == '__main__':
    unittest.main()
