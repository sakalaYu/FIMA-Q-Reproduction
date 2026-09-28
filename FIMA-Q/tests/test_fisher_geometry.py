"""Analytic CPU checks: basis geometry, Fisher metric, forward-only derivatives."""
import unittest
import torch
from utils.fisher_geometry import error_basis, projected_fisher, response_jacobian, quadratic_predictions


class GeometryTests(unittest.TestCase):
    def test_signed_basis_reconstructs_rank_two_errors(self):
        errors = torch.tensor([[1.,0.,0.],[-1.,0.,0.],[0.,2.,0.]])
        u, energies, rank = error_basis(errors,3)
        self.assertEqual(rank,2)
        torch.testing.assert_close(u @ u.T,torch.eye(2))
        torch.testing.assert_close(errors @ u.T @ u,errors)
        self.assertAlmostEqual(float(energies.sum()),6.)

    def test_zero_errors_are_not_called_rank_one(self):
        with self.assertRaises(ValueError): error_basis(torch.zeros(4,3),2)

    def test_fisher_matches_analytic_softmax_metric(self):
        jac = torch.tensor([[1.,2.],[3.,1.],[-1.,2.]],dtype=torch.float64)
        p = torch.tensor([0.2,0.3,0.5],dtype=torch.float64)
        expected = jac.T @ (torch.diag(p)-p[:,None]*p[None,:]) @ jac
        torch.testing.assert_close(projected_fisher(jac,p),expected)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(expected).min()),-1e-10)

    def test_forward_differences_match_reference_without_graph(self):
        def fn(a): return torch.stack([a[0].sin()+a[1],a[1].exp(),a.square().sum()]) / 20
        reference = response_jacobian(fn,2,torch.device('cpu'),'autograd')
        finite = response_jacobian(fn,2,torch.device('cpu'),'finite_difference',0.01)
        torch.testing.assert_close(finite,reference,rtol=1e-3,atol=1e-6)
        self.assertFalse(finite.requires_grad)

    def test_constant_logit_shifts_have_zero_fisher(self):
        v = torch.ones(3,2,dtype=torch.float64)
        f = projected_fisher(v,torch.tensor([0.2,0.3,0.5],dtype=torch.float64))
        torch.testing.assert_close(f,torch.zeros_like(f))

    def test_quadratic_predictions_preserve_correlations(self):
        pred = quadratic_predictions(torch.tensor([[2.,1.],[1.,2.]]),torch.ones(2))
        self.assertAlmostEqual(pred['full'],3.)
        self.assertAlmostEqual(pred['diagonal'],2.)


if __name__ == '__main__': unittest.main()
