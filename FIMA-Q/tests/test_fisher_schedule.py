"""Exercise the production reconstruction loop on a small CPU quantizer stub.

Checks schedule integration and monitor non-interference; the Fisher estimator
is stubbed, so these are not ImageNet/ViT accuracy tests.
"""
import ast
import logging
import math
import pathlib
import time
import unittest
try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:
    torch = None
from utils.fisher_reliability import FisherReliability


@unittest.skipIf(torch is None, 'PyTorch is required')
class ScheduleTests(unittest.TestCase):
    def run_loop(self, schedule):
        torch.manual_seed(12)

        class Rounding(nn.Module):
            def __init__(self):
                super().__init__()
                self.alpha = nn.Parameter(torch.zeros(2))
                self.soft_targets = True

            def get_soft_targets(self):
                return self.alpha.sigmoid()

        class QuantLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.w_quantizer = Rounding()
                self.a_quantizer = nn.Identity()
                self.mode = 'raw'
                self.raw_input = torch.randn(8, 2)
                self.quanted_input = self.raw_input
                self.raw_out = self.raw_input.clone()

            def forward(self, x):
                return x * (0.6 + self.w_quantizer.get_soft_targets())

        path = pathlib.Path(__file__).parents[1] / 'utils' / 'block_recon.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        recon = next(node for node in classes if node.name == 'BlockReconstructor')
        recon.bases = []
        recon.body = [node for node in recon.body if isinstance(node, ast.FunctionDef)
                      and node.name in ('reconstruct_single_block', 'set_qdrop', 'set_block_mode')]
        namespace = dict(torch=torch, nn=nn, F=F, math=math, time=time, logging=logging,
                         FisherReliability=FisherReliability, MinMaxQuantLinear=QuantLinear,
                         MinMaxQuantConv2d=type(None), MinMaxQuantMatMul=type(None))
        exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), 'exec'), namespace)
        runner = namespace['BlockReconstructor']()
        runner.metric = 'fisher_dplr'
        runner.p1 = runner.p2 = 1.
        runner.k = 3
        runner.dis_mode = 'q'
        runner.fisher_schedule = schedule
        runner.probe_interval = 1
        runner.reliability_options = dict(error_threshold=0.25, trend_threshold=0.02, patience=1)
        runner.wrap_quantizers_in_net = lambda block, name: None
        events, calls = [], []
        runner.log_fisher_event = events.append
        block = QuantLinear()

        def refresh(block, device):
            calls.append(1)
            block.raw_grad = torch.ones(1, 2)
            block.delta_out = torch.ones(1, 2)
            block.inverse_B = torch.ones(1, 1)
        runner.new_fisher_ro = refresh
        measurements = []

        def measure(block, device):
            # c is anchored on each refresh; large persistent scale drift on
            # subsequent probes must trigger, then stop at the update budget.
            measurements.append(1)
            kl = 100. ** min(len(measurements), 8)
            return [1., 2.], [1., 2.], [kl, 2 * kl]
        runner.measure_fisher_probe = measure
        runner.reconstruct_single_block('blocks.0', block, torch.device('cpu'),
                                        batch_size=4, iters=12, mode='rinp')
        return block.w_quantizer.alpha.detach().clone(), events, len(calls)

    def test_monitor_preserves_fixed_training(self):
        fixed, _, calls = self.run_loop('fixed')
        monitored, events, monitored_calls = self.run_loop('monitor')
        torch.testing.assert_close(fixed, monitored, rtol=0, atol=0)
        self.assertEqual(calls, 3)
        self.assertEqual(monitored_calls, calls)
        self.assertEqual([r['step'] for r in events if r['event'] == 'refresh'], [0, 4, 8])

    def test_adaptive_triggers_and_respects_budget(self):
        _, events, calls = self.run_loop('adaptive')
        self.assertEqual(calls, 3)
        self.assertEqual([r['step'] for r in events if r['event'] == 'refresh'], [0, 1, 2])
        self.assertTrue(any(r['trigger'] and not r['refresh'] for r in events if r['event'] == 'probe'))


if __name__ == '__main__':
    unittest.main()
