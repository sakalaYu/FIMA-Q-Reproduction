import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.semantic_probe_kd import SemanticProbeKD


class ProbeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.x = torch.randn(96, 8)
        self.logits = self.x @ torch.randn(8, 6)
        self.prob = (self.logits / 20).softmax(-1)
        self.options = dict(mode='reliable', weight=0.05, temperature=2., ridge=0.01, holdout=16)

    def make(self, **overrides):
        return SemanticProbeKD(self.x, self.prob, 20., dict(self.options, **overrides), torch.device('cpu'))

    def test_fit_generalizes_and_does_not_consume_rng(self):
        state = torch.random.get_rng_state().clone()
        probe = self.make()
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertGreater(probe.reliability, 0.95)
        self.assertLess(probe.metadata['teacher_probe_kl'], probe.metadata['teacher_prior_kl'])
        self.assertFalse(probe.weight.requires_grad)

    def test_identical_features_are_zero_and_perturbations_receive_gradients(self):
        probe = self.make()
        indices = torch.arange(16)
        self.assertLess(float(probe.loss(self.x[:16], indices, 60, 100)), 1e-4)
        student = (self.x[:16] + torch.randn(16, 8) * 0.1).requires_grad_()
        loss = probe.loss(student, indices, 60, 100)
        loss.backward()
        self.assertGreater(float(student.grad.abs().sum()), 0)
        self.assertGreater(float(loss), 0)

    def test_uniform_and_reliable_differ_only_by_skill_weight(self):
        uniform, reliable = self.make(mode='uniform'), self.make()
        student = self.x[:16] + 0.1
        indices = torch.arange(16)
        a = uniform.loss(student, indices, 60, 100)
        b = reliable.loss(student, indices, 60, 100)
        self.assertTrue(torch.allclose(b, a * reliable.reliability))
        self.assertTrue(torch.equal(uniform.weight, reliable.weight))

    def test_constant_teacher_has_no_reliable_kd_and_warmup_is_inactive(self):
        p = torch.full((96, 6), 1 / 6)
        probe = SemanticProbeKD(self.x, p, 20., self.options, torch.device('cpu'))
        self.assertEqual(probe.gain, 0)
        probe = self.make()
        self.assertEqual(float(probe.loss(self.x[:16], torch.arange(16), 20, 100)), 0)
        self.assertEqual(probe.active_steps, 0)

    def test_gradient_reaches_real_adaround_parameters(self):
        from test_qk_joint import linear
        from quantizers.adaround import AdaRoundQuantizer
        layer = linear(8, 8)
        layer.mode = 'quant_forward'
        layer.w_quantizer = AdaRoundQuantizer(layer.w_quantizer, layer.weight.view(1, 8, 8))
        layer.w_quantizer.soft_targets = True
        probe = self.make(mode='uniform')
        loss = probe.loss(layer(self.x[:16]), torch.arange(16), 60, 100)
        loss.backward()
        self.assertGreater(float(layer.w_quantizer.alpha.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
