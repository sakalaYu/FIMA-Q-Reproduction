"""Numerical regressions for the independent method integrations."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_endpoint_numerics as endpoint
from test_qk_joint import Model, Block, mode
from utils.lsvit_adapter import ls_statistics, finish_statistics, ls_terms
from utils.method_adaptations import (HeadAdaLog, attach_method_quantizers, erq_quantize,
                                     prepare_data, kd_refine, adalog_refine, erq_refine)


class MethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self):
        torch.manual_seed(13)
        student = Model().eval()
        seq = student.blocks[0].mlp
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1, self.act, self.fc2 = seq
            def forward(self, x):
                return self.fc2(self.act(self.fc1(x)))
        student.blocks[0].mlp = MLP()
        teacher = copy.deepcopy(student)
        mode(teacher, 'raw')
        mode(student, 'quant_forward')
        x = torch.randn(1024, 5, 8)
        loader = DataLoader(TensorDataset(x, torch.zeros(len(x))), batch_size=32)
        return student, teacher, loader

    def test_lsvit_streaming_matches_dense_formula(self):
        torch.manual_seed(3)
        g, d = torch.rand(12, 5, 8).double(), torch.rand(12, 5, 8).double()
        left, right = ls_statistics(g[:5], d[:5]), ls_statistics(g[5:], d[5:])
        diag, root = finish_statistics(*(a + b for a, b in zip(left, right)))
        gf, df = g.flatten(1), d.flatten(1)
        val = (gf * df).sum(1)
        self.assertTrue(torch.allclose(diag.double(), (gf * df).mean(0) / df.square().mean(0), atol=1e-6))
        expected = (gf * val.sqrt().unsqueeze(1)).mean(0) / val.mean()
        self.assertTrue(torch.allclose(root.double(), expected, atol=1e-6))
        pred = torch.randn(4, 5, 8, requires_grad=True)
        sum(ls_terms(pred, torch.zeros_like(pred), diag, root)).backward()
        self.assertTrue(bool(torch.isfinite(pred.grad).all()))

    def test_adalog_has_at_most_eight_values_and_restores_state(self):
        q = HeadAdaLog(torch.ones(1, 2, 5, 1), torch.tensor([37, 74]).view(1, 2, 1, 1))
        x = torch.logspace(-8, 0, 1000).view(1, 1, 1, -1).expand(1, 2, 5, -1)
        y = q(x)
        for head in range(2):
            self.assertLessEqual(len(y[:, head, 0].unique()), 8)
        student, _, _ = self.fixture()
        q.register_buffer('adalog_q', q.q)
        student.blocks[0].attn.matmul2.A_quantizer = q
        state = student.state_dict()
        other, _, _ = self.fixture()
        attach_method_quantizers(other, state)
        other.load_state_dict(state, strict=True)
        self.assertTrue(torch.equal(other.blocks[0].attn.matmul2.A_quantizer(x), y))

    def test_erq_keeps_the_original_three_bit_grid(self):
        torch.manual_seed(8)
        w = torch.randn(4, 16)
        x = torch.randn(40, 16)
        gram = x.T @ x / len(x)
        scale, zero = torch.full((4, 1), 0.4), torch.full((4, 1), 3.0)
        q = erq_quantize(w, gram, scale, zero)
        codes = q / scale + zero
        self.assertTrue(torch.allclose(codes, codes.round(), atol=1e-6))
        self.assertTrue(bool(((codes >= 0) & (codes <= 7)).all()))

    def test_all_three_post_methods_execute_on_real_quantizers(self):
        for method in ('kd', 'adalog', 'erq'):
            with self.subTest(method=method):
                student, teacher, loader = self.fixture()
                if method == 'kd':
                    student.blocks[0].attn.matmul1.A_quantizer.scale.data[:, 0].neg_()
                images, logits, fit, gate, targets = prepare_data(student, teacher, loader)
                if method == 'kd':
                    weights = [(m.weight.detach().clone(), m) for m in student.modules() if hasattr(m, 'w_quantizer')]
                    result = kd_refine(student, images, logits, gate, targets, 2, torch.device('cpu'))
                    self.assertTrue(all(torch.equal(w, m.weight) for w, m in weights))
                    self.assertFalse(any('parametrizations' in k for k in student.state_dict()))
                else:
                    with torch.no_grad():
                        result = (adalog_refine(student, fit, gate, targets, torch.device('cpu')) if method == 'adalog'
                                  else erq_refine(student, teacher, fit, gate, targets, torch.device('cpu')))
                self.assertLessEqual(result['gate_kl_after'], result['gate_kl_before'])
                self.assertTrue(bool(torch.isfinite(student(images[:4])).all()))

    def test_lsvit_full_block_reconstruction_uses_real_gradients(self):
        cls = endpoint.EndpointNumerics().reconstructor_class()
        dummy = type('UnusedTimmType', (nn.Module,), {})
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
        model = Mini().eval()
        x = torch.randn(12, 5, 8)
        loader = DataLoader(TensorDataset(x, torch.zeros(12)), batch_size=4)
        for module in model.modules():
            if hasattr(module, 'w_quantizer'):
                module.w_quantizer.zero_point.data.round_()
        qk = dict(variant='C', reconstruction_loss='mse', protocol='v2_integer_zp',
                  steps=2, batch_size=4, holdout=4, lr=0.003, eval_every=1, seed=3407)
        recon = cls(model, 4, loader, metric='lsvit', k=1, iters=2, qk_joint_options=qk)
        recon.reconstruct_model(quant_act=True, mode='qdrop', drop_prob=0.5, keep_gpu=False)
        self.assertTrue(bool(torch.isfinite(model(x)).all()))
        self.assertIsNone(model.blocks[0].ls_diagonal)


if __name__ == '__main__':
    unittest.main()
