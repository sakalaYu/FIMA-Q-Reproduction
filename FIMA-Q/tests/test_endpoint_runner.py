"""Prevent rank aliasing, incomplete-run reuse and mismatched gain reports."""
import copy
import argparse
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_endpoint_experiments as runner


class EndpointRunnerTests(unittest.TestCase):
    def setUp(self):
        self.args = runner.get_parser().parse_args([])

    def result(self, task):
        payload = runner.identity(task, self.args)
        return dict(status='complete', launch_identity=payload, task=task,
                    code_sha256=runner.code_fingerprint(), fp_model_sha256='model',
                    cfg=dict(k=10 if task == 'rank10' else 5, recon_iters=20000),
                    calibration_hashes=[['128', 'a', 'b'], ['1024', 'c', 'd']],
                    qk_blocks=[dict(block=f'blocks.{i}') for i in range(12)],
                    endpoint=dict(blocks=[dict(block=f'blocks.{i}') for i in range(12)]),
                    qwt=dict(blocks=[dict(block=f'blocks.{i}') for i in range(12)]),
                    final=dict(top1=65.0, top5=87.0, loss=1.4),
                    before_qwt=dict(top1=64.7, top5=87.0, loss=1.4))

    def test_rank10_reaches_actual_cli_and_other_tasks_stay_at_five(self):
        source = (runner.ROOT / 'test_quant.py').read_text(encoding='utf-8')
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'get_args_parser')
        namespace = dict(argparse=argparse)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual test_quant parser>', 'exec'), namespace)
        parser = argparse.ArgumentParser(parents=[namespace['get_args_parser']()])
        for task in runner.TASKS:
            cmd = runner.command(task, self.args)
            self.assertEqual(cmd.count('--k'), 1)
            self.assertEqual(cmd[cmd.index('--k') + 1], '10' if task == 'rank10' else '5')
            self.assertEqual(cmd[cmd.index('--qk-joint-reconstruction-loss') + 1], 'mse')
            parsed = parser.parse_args(cmd[3:])
            self.assertEqual(parsed.endpoint_task, task)
            self.assertEqual(parsed.qwt, task == 'qwt')
        self.assertEqual(runner.qk.base.RECIPE['k'], 5)

    def test_result_requires_complete_blocks_metrics_and_matching_identity(self):
        result = self.result('hard-repair')
        payload = result['launch_identity']
        self.assertTrue(runner.valid_result(result, 'hard-repair', self.args, payload))
        for key, value in (('status', 'smoke_complete'), ('qk_blocks', [{}] * 11),
                           ('endpoint', dict(blocks=[{}] * 11)), ('final', dict(top1=float('nan')))):
            bad = copy.deepcopy(result)
            bad[key] = value
            self.assertFalse(runner.valid_result(bad, 'hard-repair', self.args, payload))
        bad = copy.deepcopy(result)
        bad['launch_identity']['task'] = 'soft-control'
        self.assertFalse(runner.valid_result(bad, 'hard-repair', self.args, payload))

    def test_changed_hyperparameter_changes_resume_identity(self):
        before = runner.run_name('hard-repair', self.args)
        self.args.scale_bound = 1.1
        self.assertNotEqual(before, runner.run_name('hard-repair', self.args))

    def test_only_requested_rank_difference_is_allowed_in_comparison(self):
        left, right = self.result('rank10'), self.result('qwt')
        self.assertFalse(runner.comparable(left, right))
        self.assertTrue(runner.comparable(left, right, allow_rank=True))
        left['calibration_hashes'][1][2] = 'other-augmentation'
        self.assertFalse(runner.comparable(left, right, allow_rank=True))

    def test_smoke_has_distinct_identity_and_cannot_replace_full_run(self):
        before = runner.run_name('qwt', self.args)
        result = self.result('qwt')
        self.args.smoke = True
        self.assertNotEqual(before, runner.run_name('qwt', self.args))
        self.assertFalse(runner.valid_result(result, 'qwt', self.args, runner.identity('qwt', self.args)))

    def test_summary_reuses_completed_pre_fix_qwt_after_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(runner, 'ROOT', root), patch.object(runner, 'code_fingerprint', return_value='fixed'):
                for task in ('qwt', 'hard-repair'):
                    result = self.result(task)
                    payload = result['launch_identity']
                    if task == 'qwt':
                        runner.set_option(payload['command'], '--endpoint-code-sha', runner.PRE_FIX_CODE_SHA)
                        result['code_sha256'] = runner.PRE_FIX_CODE_SHA
                    result['run_id'] = '20261103_' + runner.run_name(task, self.args, payload)
                    folder = root / 'checkpoints' / 'quant_result' / result['run_id']
                    folder.mkdir(parents=True)
                    (folder / 'endpoint_result.json').write_text(json.dumps(result), encoding='utf-8')

                self.assertIsNone(runner.summarize(self.args, write=False)['current_C_MSE'])
                self.args.include_pre_fix = True
                report = runner.summarize(self.args, write=False)
                self.assertEqual(report['complete_count'], 2)
                self.assertEqual(report['current_C_MSE']['top1'], 64.7)
                self.assertAlmostEqual(report['rows'][0]['delta_current_C_MSE_pp'], 0.3)


if __name__ == '__main__':
    unittest.main()
