"""CPU checks of the real probe methods without requiring timm/ImageNet.

AST loading isolates the production diagnostic methods from timm imports;
these tests do not claim end-to-end ViT reconstruction coverage.
"""
import ast
import pathlib
import unittest
from contextlib import contextmanager
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is required for CPU probe tests')
class ProbeTests(unittest.TestCase):
    def setUp(self):
        path = pathlib.Path(__file__).parents[1] / 'utils' / 'block_recon.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        source = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BlockReconstructor')
        selected = ('probe_state', 'measure_fisher_probe', 'set_block_mode')
        source.bases = []
        source.body = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in selected]
        namespace = dict(torch=torch, F=F, contextmanager=contextmanager)
        exec(compile(ast.Module(body=[source], type_ignores=[]), str(path), 'exec'), namespace)
        self.runner = namespace['BlockReconstructor']()

        class ToyBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mode = 'quant_forward'
                self.drop_prob = 0.5
                self.fail = False

            def forward(self, x):
                if self.fail:
                    raise RuntimeError('test failure')
                return x if self.mode == 'raw' else x * 0.7

        self.block = ToyBlock()
        self.runner.model = torch.nn.Sequential(self.block)
        self.runner.temperature = 20
        self.runner.probe_reference = None
        self.runner.probe_batches = [(torch.tensor([[1., 2., -1.], [2., -1., 1.]]), None),
                                     (torch.tensor([[3., 1., -2.]]), None)]
        self.block.raw_grad = torch.tensor([[0.5, 1., 2.]])
        self.block.inverse_B = torch.tensor([[2.]])

    def test_probe_matches_explicit_loss_and_restores_state(self):
        state = torch.random.get_rng_state().clone()
        low, diag, kl = self.runner.measure_fisher_probe(self.block, torch.device('cpu'))
        x = torch.cat([batch[0] for batch in self.runner.probe_batches])
        e = (x * 0.7 - x).abs()
        expected_low = (e @ self.block.raw_grad.T).square().flatten() * 2
        expected_diag = (e.square() * self.block.raw_grad[0]).mean(-1)
        expected_kl = F.kl_div(F.log_softmax(x.double() * 0.7 / 20, -1),
                              F.softmax(x.double() / 20, -1), reduction='none').sum(-1)
        torch.testing.assert_close(torch.tensor(low), expected_low)
        torch.testing.assert_close(torch.tensor(diag), expected_diag)
        torch.testing.assert_close(torch.tensor(kl, dtype=torch.float64), expected_kl,
                                   atol=1e-9, rtol=1e-4)
        self.assertEqual(self.block.mode, 'quant_forward')
        self.assertEqual(self.block.drop_prob, 0.5)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertFalse(self.block._forward_hooks)
        self.assertEqual((low, diag, kl), self.runner.measure_fisher_probe(self.block, torch.device('cpu')))

    def test_exception_restores_modes_and_removes_hook(self):
        self.block.fail = True
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            self.runner.measure_fisher_probe(self.block, torch.device('cpu'))
        self.assertEqual(self.block.mode, 'quant_forward')
        self.assertEqual(self.block.drop_prob, 0.5)
        self.assertFalse(self.block._forward_hooks)


if __name__ == '__main__':
    unittest.main()
