"""B/C zero-point transition and rotation-only loss checks on real quantizers."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_qk_joint import QKJointTests, Reconstructor, QKParameters, AdaRoundQuantizer, calibrate_qk_block


class RepairNumerics(unittest.TestCase):
    def setUp(self):
        fixture = QKJointTests()
        fixture.setUp()
        self.block, self.model, self.x = fixture.block, fixture.model, fixture.x

    def test_b_and_c_preserve_integer_grid_through_adaround(self):
        for variant in ("B", "C"):
            with self.subTest(variant=variant), torch.no_grad():
                block = copy.deepcopy(self.block)
                block.attn.qkv.w_quantizer.zero_point.round_()
                fit = QKParameters(block.attn, variant)
                if variant == "C": fit.skew.normal_(0, 0.08)
                fit.log_diag.uniform_(-0.2, 0.2)
                fit.fold_into(block.attn)
                layer = block.attn.qkv
                weight = layer.weight.view(3, 8, 8)
                before = layer.w_quantizer(weight)
                adaround = AdaRoundQuantizer(layer.w_quantizer, weight)
                adaround.soft_targets = False
                self.assertTrue(torch.allclose(before, adaround(weight), atol=1e-7, rtol=1e-7))

    def test_c_mse_has_no_dplr_anchor_queries(self):
        recon = Reconstructor(self.model, self.x)
        with patch("utils.qk_joint.estimate_dplr", side_effect=AssertionError("unexpected Fisher query")):
            result = calibrate_qk_block(recon, "blocks.0", self.block, torch.device("cpu"),
                dict(variant="C", reconstruction_loss="mse", steps=3, holdout=4,
                     batch_size=4, lr=0.003, eval_every=1, seed=3407))
        self.assertEqual(result["reconstruction_loss"], "mse")
        self.assertEqual(result["fisher_rank"], 0)


if __name__ == "__main__": unittest.main()
