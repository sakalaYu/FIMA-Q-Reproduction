import unittest

import torch

from utils.robust_fisher import (balanced_indices, fisher_sample_terms,
                                 group_robust_loss, make_groups)


class RobustFisherTests(unittest.TestCase):
    def test_group_assignment_and_balanced_sampling(self):
        probability = torch.tensor([
            [.99, .01], [.95, .05], [.80, .20], [.70, .30],
            [.60, .40], [.55, .45], [.52, .48], [.50, .50],
        ])
        quant_error = torch.tensor([1., 3., 2., 4., 1., 3., 2., 4.])
        groups = make_groups(probability, quant_error)
        self.assertEqual(torch.bincount(groups, minlength=4).tolist(), [2, 2, 2, 2])
        torch.manual_seed(1)
        chosen = balanced_indices(groups, 12)
        self.assertEqual(torch.bincount(groups[chosen], minlength=4).tolist(), [3, 3, 3, 3])

    def test_sample_terms_match_existing_dplr_reduction(self):
        prediction = torch.tensor([[1., -2., 3.], [2., 1., -1.]], requires_grad=True)
        target = torch.zeros_like(prediction)
        gradient = torch.tensor([[.2, .3, .4], [.4, .2, .1]])
        inverse_b = torch.tensor([[2., .1], [.1, 1.]])
        low_rank, diagonal = fisher_sample_terms(prediction, target, gradient, inverse_b)
        error = (prediction - target).abs()
        a = error.unsqueeze(1) @ gradient.transpose(0, 1)
        old_low_rank = (a @ inverse_b @ a.transpose(1, 2)).mean()
        old_diagonal = (error.square() * gradient.mean(dim=0)).mean()
        self.assertTrue(torch.allclose(low_rank.mean(), old_low_rank))
        self.assertTrue(torch.allclose(diagonal.mean(), old_diagonal))

    def test_robust_objective_focuses_on_worse_group_and_backpropagates(self):
        sample_loss = torch.tensor([1., 1., 3., 3.], requires_grad=True)
        groups = torch.tensor([0, 0, 1, 1])
        erm = group_robust_loss(sample_loss, groups, beta=0)
        robust = group_robust_loss(sample_loss, groups, beta=1, temperature=.5)
        self.assertAlmostEqual(erm.item(), 2.)
        self.assertGreater(robust.item(), erm.item())
        robust.backward()
        self.assertGreater(sample_loss.grad[2].item(), sample_loss.grad[0].item())


if __name__ == '__main__':
    unittest.main()
