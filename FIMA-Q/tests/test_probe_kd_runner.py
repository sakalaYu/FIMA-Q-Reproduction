from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_probe_kd_experiments as runner
from test_headwise_experiments import log_text


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'logs').mkdir()
        self.patch = patch.object(runner, 'ROOT', self.root)
        self.patch.start()
        self.hash_patch = patch.object(runner, 'code_fingerprint', return_value='test-source')
        self.hash_patch.start()
        self.args = runner.get_parser().parse_args([])

    def tearDown(self):
        self.patch.stop()
        self.hash_patch.stop()
        self.temp.cleanup()

    def add(self, blocks=12):
        flags = dict(runner.qk.DEFAULTS, **runner.PARAMS,
                     qk_joint_variant='C', qk_joint_protocol='v2_integer_zp',
                     qk_joint_reconstruction_loss='mse', probe_kd='reliable',
                     probe_kd_protocol=runner.PROTOCOL, num_workers=8)
        flags['probe_kd_code_sha'] = runner.code_fingerprint()
        extra = ', '.join(f'{k}={v!r}' for k, v in flags.items()) + ', '
        text = log_text(bits=3, rotation=True).replace('Namespace(', 'Namespace(' + extra, 1)
        for i in range(blocks):
            text += '\nProbe KD result: ' + json.dumps(dict(block=f'blocks.{i}', reliability=0.5, active_steps=16000))
            text += '\nQK post-round result: ' + json.dumps(dict(block=f'blocks.{i}', accepted=True))
        (self.root / 'logs/001_probekd.log').write_text(text, encoding='utf-8')

    def test_three_jobs_keep_mse_prestage_and_full_dplr_recipe(self):
        self.assertEqual(set(runner.JOBS), {('C', 'uniform'), ('C', 'reliable'), ('B', 'reliable')})
        names = set()
        for variant, mode in runner.JOBS:
            cmd = runner.command(variant, mode, self.args)
            for flag, value in (('--optim-metric', 'fisher_dplr'), ('--recon-iters', '20000'),
                                ('--qk-joint-reconstruction-loss', 'mse'), ('--probe-kd', mode)):
                self.assertEqual(cmd[cmd.index(flag) + 1], value)
            names.add(runner.run_name(variant, mode, self.args))
        self.assertEqual(len(names), 3)

    def test_incomplete_probe_reports_are_not_resumed(self):
        self.add(blocks=11)
        self.assertEqual(runner.available_runs(), [])

    def test_resume_requires_same_probe_configuration(self):
        self.add()
        runs = runner.available_runs()
        self.assertIsNotNone(runner.latest(runs, 'C', 'reliable', self.args))
        self.assertIsNone(runner.latest(runs, 'C', 'uniform', self.args))
        self.args.probe_kd_temperature = 3
        self.assertIsNone(runner.latest(runs, 'C', 'reliable', self.args))


if __name__ == '__main__':
    unittest.main()
