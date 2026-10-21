"""Check that signed task alignment keeps helpful relations and rejects conflicts."""

import sys
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.attention_relation import AttentionRelation, ALIGNMENT_PROTOCOL


class FakeAttention:
    scale = 1.0
    matmul1 = torch.nn.Identity()


class FakeBlock:
    attn = FakeAttention()


class AttentionAlignmentTests(unittest.TestCase):
    def test_alignment_keeps_only_task_consistent_head(self):
        relation = AttentionRelation(FakeBlock(), "blocks.0", dict(
            mode="aligned", protocol=ALIGNMENT_PROTOCOL, rows=1,
            weight=0.1, start=0.2, full=0.6))
        relation.indices = torch.tensor([0])
        teacher = torch.tensor([0.8, 0.2]).log().view(1, 1, 1, 2)
        relation.teacher = teacher.expand(2, 2, 1, 2).clone()
        relation.weights = torch.ones(2, 1)
        student = torch.tensor([0.2, 0.8]).log().view(1, 1, 1, 2)
        scores = student.expand(2, 2, 1, 2).clone().requires_grad_(True)
        relation.begin_refresh()
        gradient = torch.tensor([-0.6, 0.6]).view(1, 1, 1, 2).expand_as(scores).clone()
        gradient[:, 1].neg_()
        forward = FakeBlock.attn.matmul1.register_forward_hook(relation.refresh_score_forward)
        backward = FakeBlock.attn.matmul1.register_full_backward_hook(relation.score_gradient)
        try:
            output = FakeBlock.attn.matmul1(scores)
            (output * gradient).sum().backward()
        finally:
            forward.remove()
            backward.remove()
        relation.end_refresh()
        self.assertTrue(torch.allclose(relation.weights[:, 0], torch.tensor([2.0, 0.0])))
        self.assertAlmostEqual(relation.alignment_positive_fraction, 0.5)
        self.assertEqual(relation.gradient_refreshes, 1)


if __name__ == "__main__":
    unittest.main()
