"""CPU checks for interaction MR's quantization, attribution and baseline reuse."""

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mr_interaction_impl as impl
from utils import lsvit_mr as mr


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.ModuleDict({'mlp': MLP()}) for _ in range(12)])
        # ModuleDict exposes children as attributes, as required by MR.


class InteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)
        self.teacher = MLP().eval()
        self.x = torch.randn(16, 3, 4)
        with torch.no_grad():
            self.y = self.teacher(self.x)

    def test_quantizer_matches_integer_grid_and_has_gradients(self):
        x = torch.randn(4, 8, requires_grad=True)
        q = impl.make_quantizer(x, per_row=True)
        self.assertTrue(torch.equal(q.zero_point, q.zero_point.round()))
        hard = ((x / q.scale).round() + q.zero_point).clamp(0, 7).sub(q.zero_point).mul(q.scale)
        self.assertTrue(torch.equal(q(x), hard))
        q(x).square().mean().backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertGreater(float(x.grad.abs().sum()), 0)

    def test_interaction_weights_are_bounded_and_detached(self):
        errors = tuple(torch.tensor([v, 1.], requires_grad=True) for v in (0., 1., 1., 4.))
        weights, delta = impl.interaction_weights(errors)
        self.assertFalse(weights.requires_grad)
        self.assertTrue(bool(((weights >= 1) & (weights <= 2)).all()))
        self.assertGreater(float(weights[0]), 1)
        self.assertEqual(float(weights[1]), 1)
        self.assertGreater(float(delta[0]), 0)

    def test_four_paths_and_zero_interaction_without_quantization(self):
        student = copy.deepcopy(self.teacher)
        student.act = nn.ReLU()
        paths = impl.four_paths(student, self.x, [nn.Identity()] * 4)
        self.assertTrue(all(torch.equal(paths[0], y) for y in paths))
        weights, delta = impl.interaction_weights(impl.token_errors(paths, self.y))
        self.assertTrue(torch.equal(weights, torch.ones_like(weights)))
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))

    def test_continuation_preserves_baseline_and_common_start(self):
        reference = mr.start_fit(self.teacher, 20, 3407)
        mr.train_fit(reference, self.x, self.y, 'cpu', 20, 1.)
        anchors, states = [], []
        for variant in ('baseline', 'joint', 'interaction'):
            fit = mr.start_fit(self.teacher, 20, 3407)
            anchors.append(impl.continue_fit(mr.train_fit, fit, self.x, self.y, 'cpu', 20, 1., variant))
            states.append(fit[0].state_dict())
            self.assertTrue(all(bool(torch.isfinite(v).all()) for v in states[-1].values()))
            self.assertFalse(fit[0].fc2._forward_pre_hooks)
        self.assertEqual(len(set(anchors)), 1)
        self.assertTrue(all(torch.equal(v, reference[0].state_dict()[k]) for k, v in states[0].items()))
        self.assertTrue(any(not torch.equal(states[1][k], states[2][k]) for k in states[1]))

    def test_residual_baseline_pipeline_is_unchanged(self):
        model = ToyModel().eval()
        original_train = mr.train_fit
        original_fit = mr.fit_with_residual_budget

        def fast_train(fit, x, y, device, steps, clip):
            # 1000, 2000, 3000 and their halves map to integer update counts.
            return original_train(fit, x, y, device, steps // 250, clip)

        def features(model, mlp, loader, device):
            with torch.no_grad():
                return self.x, mlp(self.x)

        with patch.object(mr, 'teacher_features', features), patch.object(mr, 'train_fit', fast_train):
            expected, _ = original_fit(model, None, 'cpu', 3407, 4, 'residual')
            actual, reports = impl.fit_residual(model, None, 'cpu', 3407, 4, 'residual', 'baseline', original_fit)
            self.assertIs(mr.train_fit, fast_train)
        self.assertTrue(all(torch.equal(expected[i][k], actual[i][k])
                            for i in range(12) for k in expected[i]))
        self.assertEqual(sum(r['steps'] for r in reports), 36000)
        self.assertTrue(all('joint_w3a3_nmse' in r['interaction_diagnostics'] for r in reports))


if __name__ == '__main__':
    unittest.main()
