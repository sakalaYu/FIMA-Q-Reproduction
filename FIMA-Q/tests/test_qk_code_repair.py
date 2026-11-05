"""Check real quantization gradients, integer constraints, rollback and reload."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_method_adaptations as methods
from utils.method_adaptations import prepare_data
from utils.qk_code_repair import propose_codes, refine_qk_codes


class CodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_integer_moves_respect_qk_mask_and_budget(self):
        scale, zero = torch.ones(12, 1) * 0.3, torch.ones(12, 1) * 3
        codes = torch.arange(48).reshape(12, 4) % 8
        w = (codes - zero) * scale
        candidate, count = propose_codes(w, torch.ones_like(w), scale, zero, 4, 7)
        new_codes = candidate / scale + zero
        self.assertEqual(count, 7)
        self.assertTrue(torch.allclose(new_codes, new_codes.round(), atol=1e-6))
        self.assertTrue(bool(((new_codes >= -1e-6) & (new_codes <= 7 + 1e-6)).all()))
        self.assertLessEqual(float((new_codes - codes).abs().max()), 1.000001)
        self.assertTrue(torch.equal(candidate[8:], w[8:]))
        self.assertEqual(int((candidate != w).sum()), 7)

    def test_zero_gradient_is_exact_identity(self):
        w = torch.randn(12, 4)
        result, count = propose_codes(w, torch.zeros_like(w), torch.ones(12, 1),
                                     torch.ones(12, 1) * 3, 4, 128)
        self.assertEqual(count, 0)
        self.assertTrue(torch.equal(result, w))

    def fixture(self):
        model, teacher, loader = methods.MethodTests().fixture()
        return model, prepare_data(model, teacher, loader)

    def test_forced_rejection_restores_entire_state_and_rng(self):
        model, (images, logits, _, gate, targets) = self.fixture()
        initial = copy.deepcopy(model.state_dict())
        flags = [p.requires_grad for p in model.parameters()]
        rng = torch.random.get_rng_state().clone()
        with patch('utils.qk_code_repair.teacher_kl', return_value=1.0):
            report = refine_qk_codes(model, images, logits, gate, targets, torch.device('cpu'))
        self.assertEqual(report['changed_entries'], 0)
        self.assertTrue(all(torch.equal(initial[k], v) for k, v in model.state_dict().items()))
        self.assertEqual(flags, [p.requires_grad for p in model.parameters()])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_search_preserves_non_qk_state_and_serialization(self):
        model, (images, logits, _, gate, targets) = self.fixture()
        before = copy.deepcopy(model.state_dict())
        result = refine_qk_codes(model, images, logits, gate, targets, torch.device('cpu'))
        self.assertLessEqual(result['gate_kl_after'], result['gate_kl_before'])
        for key, value in model.state_dict().items():
            if key.endswith('attn.qkv.weight'):
                self.assertTrue(torch.equal(value[16:], before[key][16:]))
            else:
                self.assertTrue(torch.equal(value, before[key]), key)
        other = copy.deepcopy(model)
        other.load_state_dict(model.state_dict(), strict=True)
        self.assertTrue(torch.equal(model(images[:4]), other(images[:4])))

    def test_accepted_move_commits_and_matches_inference(self):
        from utils.hard_scale_repair import quantization_path
        model, (images, logits, _, gate, targets) = self.fixture()
        with patch('utils.qk_code_repair.teacher_kl', side_effect=[1.0, 0.5]):
            result = refine_qk_codes(model, images, logits, gate, targets, torch.device('cpu'))
        self.assertGreater(result['changed_entries'], 0)
        self.assertEqual(result['gate_kl_after'], 0.5)
        inference = model(images[:4]).detach()
        with quantization_path(model, False, 1.0):
            ste = model(images[:4]).detach()
        self.assertTrue(torch.equal(inference, ste))


if __name__ == '__main__':
    unittest.main()
