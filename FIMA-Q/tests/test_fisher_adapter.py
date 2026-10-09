import unittest

import torch
from torch import nn

from utils.fisher_adapter import CorrectedBranch, normalized_feature_loss


class FisherAdapterTests(unittest.TestCase):
    def test_zero_initialized_adapter_preserves_branch(self):
        torch.manual_seed(3)
        base = nn.Linear(8, 8)
        branch = CorrectedBranch(base, width=8, rank=2)
        x = torch.randn(2, 4, 8)
        self.assertTrue(torch.equal(branch(x), base(x)))

    def test_adapter_receives_gradient_with_frozen_base(self):
        torch.manual_seed(5)
        branch = CorrectedBranch(nn.Linear(8, 8), width=8, rank=2)
        for parameter in branch.base.parameters():
            parameter.requires_grad_(False)
        prediction = branch(torch.randn(2, 4, 8))
        prediction.square().mean().backward()
        self.assertIsNone(branch.base.weight.grad)
        self.assertGreater(branch.adapter.up.weight.grad.abs().sum().item(), 0)

    def test_fisher_channel_weights_change_feature_objective(self):
        prediction = {'attn': torch.tensor([[[2., 1.]]])}
        target = {'attn': torch.tensor([[[1., 1.]]])}
        plain = normalized_feature_loss(prediction, target)
        weighted = normalized_feature_loss(
            prediction, target, {'attn': torch.tensor([1.8, .2])})
        self.assertGreater(weighted.item(), plain.item())


if __name__ == '__main__':
    unittest.main()
