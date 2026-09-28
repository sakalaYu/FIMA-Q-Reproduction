"""Analytic checks for the fixed per-sample forward-secant loss."""
import unittest
import torch
from utils.forward_secant import forward_secant_normalizers, forward_secant_terms


class ForwardSecantTests(unittest.TestCase):
    def test_anchor_error_recovers_mean_forward_kl(self):
        target = torch.zeros(2, 2, 3)
        direction = torch.tensor([
            [[1., -2., 0.], [0.5, 1., -1.]],
            [[-1., 0., 2.], [1., -0.5, 0.]],
        ])
        kl = torch.tensor([0.2, 0.8])
        secant, _ = forward_secant_terms(target + direction, target, direction, kl)
        self.assertAlmostEqual(float(secant), float(kl.mean()), places=6)

    def test_scaled_error_obeys_fixed_quadratic_secant(self):
        target = torch.zeros(2, 4)
        direction = torch.tensor([[1., -1., 2., 0.], [0.5, 1., 0., -2.]])
        kl = torch.tensor([0.3, 0.7])
        secant, _ = forward_secant_terms(target + 1.5 * direction, target, direction, kl)
        self.assertAlmostEqual(float(secant), float(2.25 * kl.mean()), places=6)

    def test_anchor_normalizers_make_both_terms_one(self):
        target = torch.zeros(2, 4)
        direction = torch.tensor([[1., -1., 2., 0.], [0.5, 1., 0., -2.]])
        kl = torch.tensor([0.3, 0.7])
        secant, residual = forward_secant_terms(target + direction, target, direction, kl)
        secant_scale, residual_scale = forward_secant_normalizers(direction, kl)
        self.assertAlmostEqual(float(secant / secant_scale), 1., places=6)
        self.assertAlmostEqual(float(residual / residual_scale), 1., places=6)

    def test_orthogonal_error_is_covered_by_fixed_mse_residual(self):
        target = torch.zeros(1, 2)
        direction = torch.tensor([[1., 0.]])
        prediction = torch.tensor([[0., 2.]], requires_grad=True)
        secant, residual = forward_secant_terms(prediction, target, direction, torch.tensor([1.]))
        self.assertEqual(float(secant), 0.)
        self.assertEqual(float(residual), 2.)
        residual.backward()
        self.assertIsNotNone(prediction.grad)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            forward_secant_terms(torch.zeros(2, 3), torch.zeros(2, 3),
                                  torch.zeros(1, 3), torch.zeros(2))


if __name__ == '__main__':
    unittest.main()
